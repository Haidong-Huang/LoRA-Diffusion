"""
专家池算法: 实现参数高效的专家池设计
- 扩散教师策略提供学习目标
- 共享主干冻结
- LoRA专家适配头
- 蒸馏训练
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.models.diffusion_teacher import DiffusionTeacherPolicy
from libero.lifelong.models.expert_policy import ExpertPolicy
from libero.lifelong.models.bc_rnn_policy import BCRNNPolicy
from libero.lifelong.metric import *
from libero.lifelong.utils import *


class ExpertPoolAlgo(Sequential):
    """
    专家池算法:
    1. 使用预训练的扩散教师策略
    2. 为每个任务/概念簇创建专家 (冻结主干 + LoRA适配头)
    3. 通过蒸馏训练专家
    """
    
    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        
        # 初始化教师策略 (扩散模型)
        # 如果提供了预训练模型路径,加载它;否则使用当前策略作为教师
        self.teacher_policy = None
        self.experts = {}  # {task_id: ExpertPolicy}
        self.current_expert = None
        
        # 保存原始policy作为fallback (用于评估时的兼容性)
        self.original_policy = self.policy
        
        # 创建一个property来动态返回当前专家 (用于评估兼容性)
        # 这样algo.policy.get_action()会自动使用当前专家
        
        # LoRA配置
        self.lora_rank = getattr(cfg.lifelong, 'lora_rank', 10)
        
        # 蒸馏损失权重
        self.distill_loss_weight = getattr(cfg.lifelong, 'distill_loss_weight', 1.0)
        self.bc_loss_weight = getattr(cfg.lifelong, 'bc_loss_weight', 1.0)
        
        # 路由权重 (用于加权蒸馏损失)
        self.use_routing = getattr(cfg.lifelong, 'use_routing', False)
    
    def _setup_teacher_policy(self):
        """设置教师策略"""
        if self.teacher_policy is None:
            # 使用当前策略作为教师 (假设已经预训练)
            # 或者从预训练模型加载
            if self.cfg.pretrain_model_path != "":
                # 加载预训练的教师策略
                teacher_cfg = self.cfg.copy()
                teacher_cfg.policy.policy_type = "bc_rnn_policy"  # 使用BCRNN作为基础
                self.teacher_policy = BCRNNPolicy(teacher_cfg, self.cfg.shape_meta)
                state_dict = torch_load_model(self.cfg.pretrain_model_path)[0]
                self.teacher_policy.load_state_dict(state_dict)
            else:
                # 使用当前策略作为教师 (需要先预训练)
                # 这里简化: 直接使用当前策略
                self.teacher_policy = self.policy
            
            # 将教师策略转换为扩散教师 (如果需要)
            # 简化: 直接使用BCRNN作为教师,不实现完整扩散
            # 实际应该使用DiffusionTeacherPolicy
    
    def start_task(self, task):
        """开始新任务时创建专家"""
        super().start_task(task)
        
        # 设置教师策略
        self._setup_teacher_policy()
        
        # 为当前任务创建专家
        if task not in self.experts:
            expert = ExpertPolicy(
                cfg=self.cfg,
                shape_meta=self.cfg.shape_meta,
                teacher_backbone=self.teacher_policy,
                expert_id=task,
                lora_rank=self.lora_rank,
            )
            expert = safe_device(expert, self.cfg.device)
            self.experts[task] = expert
        
        self.current_expert = self.experts[task]
        
        # 为了兼容evaluate_one_task_success中的algo.policy.get_action()
        # 我们将policy设置为当前专家
        self.policy = self.current_expert
        
        # 更新优化器以只优化LoRA参数
        lora_params = list(self.current_expert.lora_adapter.parameters())
        self.optimizer = eval(self.cfg.train.optimizer.name)(
            lora_params, **self.cfg.train.optimizer.kwargs
        )
        
        # 更新调度器
        self.scheduler = None
        if self.cfg.train.scheduler is not None:
            self.scheduler = eval(self.cfg.train.scheduler.name)(
                self.optimizer,
                T_max=self.cfg.train.n_epochs,
                **self.cfg.train.scheduler.kwargs,
            )
    
    def observe(self, data):
        """
        观察数据并更新专家 (蒸馏训练)
        """
        data = self.map_tensor_to_device(data)
        
        # 获取教师动作 (用于蒸馏)
        with torch.no_grad():
            # 简化: 直接使用真实动作作为教师动作
            # 实际应该使用扩散教师生成
            teacher_action = data["actions"]
            
            # 或者使用教师策略生成
            # teacher_data = self.current_expert.preprocess_input(data, train_mode=True)
            # teacher_action = self.teacher_policy.forward(teacher_data)
            # if isinstance(teacher_action, torch.distributions.Distribution):
            #     teacher_action = teacher_action.sample()
        
        # 计算路由权重 (如果使用路由)
        routing_weight = None
        if self.use_routing:
            # 简化: 使用均匀权重
            # 实际应该根据概念编码器计算 q(e|s_i)
            routing_weight = torch.ones(data["actions"].shape[0], device=self.cfg.device)
        
        # 计算损失
        self.optimizer.zero_grad()
        
        # 蒸馏损失: 专家学习教师动作
        distill_loss = self.current_expert.compute_loss(
            data,
            teacher_action=teacher_action,
            routing_weight=routing_weight,
            reduction="mean"
        )
        
        # 行为克隆损失: 专家学习真实动作 (可选)
        bc_loss = F.mse_loss(
            self.current_expert.forward(data, train_mode=True),
            data["actions"],
            reduction="mean"
        )
        
        # 总损失
        total_loss = (
            self.distill_loss_weight * distill_loss +
            self.bc_loss_weight * bc_loss
        )
        
        total_loss.backward()
        
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.current_expert.lora_adapter.parameters(),
                self.cfg.train.grad_clip
            )
        
        self.optimizer.step()
        
        return total_loss.item()
    
    def eval_observe(self, data):
        """评估观察"""
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            loss = self.current_expert.compute_loss(data, reduction="mean")
        return loss.item()
    
    def learn_one_task(self, dataset, task_id, benchmark, result_summary):
        """学习单个任务"""
        self.start_task(task_id)
        
        # 恢复对应的manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))
        
        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_expert.pth"
        )
        
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            persistent_workers=self.cfg.train.num_workers > 0,
        )
        
        prev_success_rate = -1.0
        best_state_dict = self.current_expert.lora_adapter.state_dict()
        
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []
        
        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)
        
        # 开始训练
        for epoch in range(0, self.cfg.train.n_epochs + 1):
            t0 = time.time()
            
            if epoch > 0:
                self.current_expert.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            else:
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()
            
            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
            )
            
            if epoch % self.cfg.eval.eval_every == 0:
                losses.append(training_loss)
                
                t0 = time.time()
                
                task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"
                sim_states = (
                    result_summary[task_str] if self.cfg.eval.save_sim_states else None
                )
                
                # 使用当前专家进行评估
                success_rate = evaluate_one_task_success(
                    cfg=self.cfg,
                    algo=self,
                    task=task,
                    task_emb=task_emb,
                    task_id=task_id,
                    sim_states=sim_states,
                    task_str="",
                )
                successes.append(success_rate)
                
                if prev_success_rate < success_rate:
                    torch_save_model(
                        self.current_expert.lora_adapter,
                        model_checkpoint_name,
                        cfg=self.cfg
                    )
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1
                    best_state_dict = self.current_expert.lora_adapter.state_dict()
                
                t1 = time.time()
                
                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                print(
                    f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                    + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                    flush=True,
                )
            
            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()
        
        # 加载最佳模型
        self.current_expert.lora_adapter.load_state_dict(
            torch_load_model(model_checkpoint_name)[0]
        )
        
        # 结束任务
        self.end_task(dataset, task_id, benchmark)
        
        # 返回指标
        losses = np.array(losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": losses,
            },
            auc_checkpoint_name,
        )
        
        losses[idx_at_best_succ:] = losses[idx_at_best_succ]
        successes[idx_at_best_succ:] = successes[idx_at_best_succ]
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter
    
    def get_action(self, data, task_id=None):
        """获取动作 (使用对应任务的专家)"""
        if task_id is not None and task_id in self.experts:
            expert = self.experts[task_id]
        elif self.current_expert is not None:
            expert = self.current_expert
        else:
            # 回退到默认策略
            if hasattr(self, 'policy') and self.policy is not None:
                return self.policy.get_action(data)
            else:
                raise ValueError("No expert or policy available")
        
        return expert.get_action(data)
    
    def reset(self):
        """重置所有专家"""
        for expert in self.experts.values():
            expert.reset()
