"""
专家策略: 使用冻结共享主干 + LoRA适配头
每个专家通过轻量级LoRA适配器学习任务特定的动作生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.lora_adapter import LoRAAdapter
from libero.lifelong.models.policy_head import GMMHead


class ExpertPolicy(BasePolicy):
    """
    专家策略: 
    - 共享主干 (冻结): 图像编码器 + 语言编码器 + RNN
    - LoRA适配头: 将主干特征转换为动作
    """
    
    def __init__(self, cfg, shape_meta, teacher_backbone, expert_id=0, lora_rank=10):
        """
        Args:
            cfg: 配置对象
            shape_meta: 形状元数据
            teacher_backbone: 教师策略的共享主干 (将被冻结)
            expert_id: 专家ID
            lora_rank: LoRA的秩
        """
        super().__init__(cfg, shape_meta)
        self.expert_id = expert_id
        self.lora_rank = lora_rank
        
        # 复制并冻结共享主干
        self._setup_frozen_backbone(teacher_backbone)
        
        # 创建LoRA适配头
        self._setup_lora_head(cfg, shape_meta)
        
        # 策略头 (用于计算损失,但实际输出通过LoRA)
        # 这里我们使用GMMHead来保持兼容性
        # 注意: 实际输出通过LoRA适配器,policy_head仅用于兼容性
        policy_head_kwargs = cfg.policy.policy_head.network_kwargs.copy()
        policy_head_kwargs['input_size'] = self.backbone_output_dim
        policy_head_kwargs['output_size'] = shape_meta["ac_dim"]
        if hasattr(cfg.policy.policy_head, 'loss_kwargs'):
            loss_kwargs = cfg.policy.policy_head.loss_kwargs.copy()
        else:
            loss_kwargs = {}
        self.policy_head = eval(cfg.policy.policy_head.network)(
            **loss_kwargs,
            **policy_head_kwargs
        )
    
    def _setup_frozen_backbone(self, teacher_backbone):
        """设置冻结的共享主干"""
        # 复制图像编码器
        self.image_encoders = nn.ModuleDict()
        for name, encoder_info in teacher_backbone.image_encoders.items():
            self.image_encoders[name] = {
                "input_shape": encoder_info["input_shape"],
                "encoder": encoder_info["encoder"]
            }
        
        # 复制语言编码器
        self.language_encoder = teacher_backbone.language_encoder
        
        # 复制额外编码器
        self.extra_encoder = teacher_backbone.extra_encoder
        
        # 复制RNN
        self.rnn = teacher_backbone.rnn
        self.D = teacher_backbone.D
        
        # 冻结所有主干参数
        for param in self.image_encoders.parameters():
            param.requires_grad = False
        for param in self.language_encoder.parameters():
            param.requires_grad = False
        for param in self.extra_encoder.parameters():
            param.requires_grad = False
        for param in self.rnn.parameters():
            param.requires_grad = False
        
        # 保存eval状态相关的变量
        self.eval_h0 = None
        self.eval_c0 = None
    
    def _setup_lora_head(self, cfg, shape_meta):
        """设置LoRA适配头"""
        # 计算主干输出维度
        policy_cfg = cfg.policy
        image_embed_size = policy_cfg.image_embed_size
        text_embed_size = policy_cfg.text_embed_size
        
        rnn_input_size = 0
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                rnn_input_size += image_embed_size
        rnn_input_size += text_embed_size
        rnn_input_size += self.extra_encoder.extra_low_level_feature_dim
        
        self.backbone_output_dim = self.D * policy_cfg.rnn_hidden_size
        
        # 获取教师策略的输出层权重 (如果存在)
        freeze_weight = None
        freeze_bias = None
        # 简化: 不直接使用教师策略的输出层权重
        # 实际实现中可以从教师策略的policy_head提取权重
        
        # 创建LoRA适配器
        self.lora_adapter = LoRAAdapter(
            input_dim=self.backbone_output_dim,
            output_dim=shape_meta["ac_dim"],
            rank=self.lora_rank,
            alpha=1.0,
            freeze_weight=freeze_weight,
            freeze_bias=freeze_bias,
        )
    
    def forward(self, data, train_mode=True):
        """
        前向传播: 通过冻结主干提取特征,然后通过LoRA适配头生成动作
        
        Args:
            data: 输入数据字典
            train_mode: 是否为训练模式
        Returns:
            action: (B, T, action_dim) - 专家动作
        """
        # 1. 通过冻结主干提取特征
        backbone_features = self._extract_backbone_features(data, train_mode)
        
        # 2. 通过LoRA适配头生成动作
        action = self.lora_adapter(backbone_features)
        
        # 为了兼容现有的损失计算,我们需要返回一个分布
        # 这里我们创建一个确定性分布 (实际可以使用GMM)
        # 简化: 直接返回动作,让policy_head处理分布
        return action
    
    def _extract_backbone_features(self, data, train_mode=True):
        """
        通过冻结主干提取特征
        
        Returns:
            features: (B, T, feature_dim) - 主干输出特征
        """
        # 1. 编码图像
        encoded = []
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, -1)
            encoded.append(e)
        
        # 2. 添加额外模态
        encoded.append(self.extra_encoder(data["obs"]))
        encoded = torch.cat(encoded, -1)
        
        # 3. 语言编码
        lang_h = self.language_encoder(data)
        encoded = torch.cat(
            [encoded, lang_h.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1
        )
        
        # 4. RNN编码
        if train_mode:
            h0 = torch.zeros(
                self.D * self.cfg.policy.rnn_num_layers,
                encoded.shape[0],
                self.cfg.policy.rnn_hidden_size,
            ).to(self.device)
            c0 = torch.zeros(
                self.D * self.cfg.policy.rnn_num_layers,
                encoded.shape[0],
                self.cfg.policy.rnn_hidden_size,
            ).to(self.device)
            output, (hn, cn) = self.rnn(encoded, (h0, c0))
        else:
            if self.eval_h0 is None:
                self.eval_h0 = torch.zeros(
                    self.D * self.cfg.policy.rnn_num_layers,
                    encoded.shape[0],
                    self.cfg.policy.rnn_hidden_size,
                ).to(self.device)
                self.eval_c0 = torch.zeros(
                    self.D * self.cfg.policy.rnn_num_layers,
                    encoded.shape[0],
                    self.cfg.policy.rnn_hidden_size,
                ).to(self.device)
            output, (h1, c1) = self.rnn(encoded, (self.eval_h0, self.eval_c0))
            self.eval_h0 = h1.detach()
            self.eval_c0 = c1.detach()
        
        return output  # (B, T, feature_dim)
    
    def compute_loss(self, data, teacher_action=None, routing_weight=None, reduction="mean"):
        """
        计算损失: 支持行为克隆损失和蒸馏损失
        
        Args:
            data: 输入数据
            teacher_action: (B, T, action_dim) - 教师动作 (用于蒸馏)
            routing_weight: (B,) - 路由权重 (用于加权损失)
            reduction: 损失归约方式
        Returns:
            loss: 标量损失值
        """
        data = self.preprocess_input(data, train_mode=True)
        
        # 获取专家动作
        expert_action = self.forward(data, train_mode=True)
        
        if teacher_action is not None:
            # 蒸馏损失: 最小化专家动作与教师动作的差异
            if routing_weight is not None:
                # 加权损失
                diff = (expert_action - teacher_action) ** 2
                diff = diff.mean(dim=-1)  # (B, T)
                diff = diff.mean(dim=-1)  # (B,)
                loss = (diff * routing_weight).sum() / (routing_weight.sum() + 1e-8)
            else:
                # 标准L2损失
                loss = F.mse_loss(expert_action, teacher_action, reduction=reduction)
        else:
            # 标准行为克隆损失
            # 将专家动作转换为分布
            # 简化: 使用MSE损失
            target_action = data["actions"]
            loss = F.mse_loss(expert_action, target_action, reduction=reduction)
        
        return loss
    
    def get_action(self, data):
        """获取专家动作"""
        self.eval()
        data = self.preprocess_input(data, train_mode=False)
        with torch.no_grad():
            action = self.forward(data, train_mode=False)
        return action.view(action.shape[0], -1).detach().cpu().numpy()
    
    def reset(self):
        """重置RNN状态"""
        self.eval_h0 = None
        self.eval_c0 = None
