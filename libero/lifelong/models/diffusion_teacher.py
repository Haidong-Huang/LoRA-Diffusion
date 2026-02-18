"""
扩散教师策略: 通过T步去噪过程生成动作序列
作为专家学生的教师,提供高质量的学习目标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from libero.lifelong.models.bc_rnn_policy import BCRNNPolicy


class DiffusionTeacherPolicy(BCRNNPolicy):
    """
    扩散教师策略: 基于BCRNNPolicy,添加扩散去噪过程
    
    通过T步去噪步骤生成动作,每一步都预测去噪后的动作
    """
    
    def __init__(self, cfg, shape_meta, num_diffusion_steps=100, beta_schedule='linear'):
        """
        Args:
            cfg: 配置对象
            shape_meta: 形状元数据
            num_diffusion_steps: 扩散步数 T
            beta_schedule: 噪声调度 ('linear', 'cosine')
        """
        super().__init__(cfg, shape_meta)
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_schedule = beta_schedule
        
        # 计算噪声调度
        self._setup_noise_schedule()
        
        # 冻结所有参数 (教师策略在训练专家时保持冻结)
        for param in self.parameters():
            param.requires_grad = False
    
    def _setup_noise_schedule(self):
        """设置扩散过程的噪声调度"""
        if self.beta_schedule == 'linear':
            # 线性调度
            self.betas = torch.linspace(0.0001, 0.02, self.num_diffusion_steps)
        elif self.beta_schedule == 'cosine':
            # 余弦调度
            s = 0.008
            steps = torch.arange(self.num_diffusion_steps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(((steps / self.num_diffusion_steps) + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        # 计算累积量
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 注册为buffer以便移动到设备
        self.register_buffer('betas', self.betas)
        self.register_buffer('alphas', self.alphas)
        self.register_buffer('alphas_cumprod', self.alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', self.alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', self.sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', self.sqrt_one_minus_alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程: 在时间步t添加噪声
        
        Args:
            x_start: (B, T, action_dim) - 原始动作
            t: (B,) - 时间步索引
            noise: (B, T, action_dim) - 可选的外部噪声
        Returns:
            x_t: (B, T, action_dim) - 加噪后的动作
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample_step(self, x_t, t, state_features):
        """
        单步去噪: 从x_t预测x_{t-1}
        
        Args:
            x_t: (B, T, action_dim) - 当前噪声动作
            t: (B,) - 当前时间步
            state_features: (B, T, feature_dim) - 状态特征 (来自共享主干)
        Returns:
            x_prev: (B, T, action_dim) - 去噪后的动作
            pred_x0: (B, T, action_dim) - 预测的干净动作
        """
        # 使用策略网络预测去噪后的动作
        # 这里我们需要将噪声动作和状态特征结合
        # 简化实现: 将噪声动作作为额外输入
        
        # 构建输入数据 (需要适配BCRNNPolicy的输入格式)
        # 这里我们直接使用策略头预测,但实际应该使用完整的去噪网络
        # 为了简化,我们假设策略网络可以接受噪声动作作为输入
        
        # 预测噪声 (简化实现)
        # 实际应该使用专门的去噪网络
        pred_noise = self._predict_noise(x_t, state_features, t)
        
        # 计算预测的干净动作
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.alphas[t].view(-1, 1, 1))
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        pred_x0 = sqrt_recip_alphas_t * (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise)
        
        # 计算后验方差
        posterior_variance = self.betas[t].view(-1, 1, 1) * (1.0 - self.alphas_cumprod_prev[t].view(-1, 1, 1)) / (1.0 - self.alphas_cumprod[t].view(-1, 1, 1))
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        
        # 采样 x_{t-1}
        noise = torch.randn_like(x_t) if t.min() > 0 else torch.zeros_like(x_t)
        posterior_mean = (
            torch.sqrt(self.alphas_cumprod_prev[t].view(-1, 1, 1)) * self.betas[t].view(-1, 1, 1) / (1.0 - self.alphas_cumprod[t].view(-1, 1, 1)) * pred_x0 +
            torch.sqrt(self.alphas[t].view(-1, 1, 1)) * (1.0 - self.alphas_cumprod_prev[t].view(-1, 1, 1)) / (1.0 - self.alphas_cumprod[t].view(-1, 1, 1)) * x_t
        )
        x_prev = posterior_mean + torch.exp(0.5 * posterior_log_variance_clipped) * noise
        
        return x_prev, pred_x0
    
    def _predict_noise(self, x_t, state_features, t):
        """
        预测噪声 (简化实现)
        实际应该使用专门的去噪网络,这里使用策略网络近似
        """
        # 简化: 直接使用策略网络输出作为去噪预测
        # 实际实现中应该训练一个专门的去噪网络
        # 这里我们假设可以通过某种方式将噪声动作和状态特征结合
        with torch.no_grad():
            # 使用策略网络预测 (这里需要适配输入格式)
            # 为了简化,我们返回零噪声 (实际应该训练去噪网络)
            return torch.zeros_like(x_t)
    
    def generate_action(self, data, return_all_steps=False):
        """
        通过完整扩散过程生成动作
        
        Args:
            data: 输入数据字典
            return_all_steps: 是否返回所有中间步骤
        Returns:
            action: (B, T, action_dim) - 最终生成的动作
            all_steps: (可选) 所有中间步骤的动作
        """
        self.eval()
        B = data["task_emb"].shape[0]
        T = self.cfg.data.seq_len
        action_dim = self.shape_meta["ac_dim"]
        
        # 从纯噪声开始
        x_t = torch.randn(B, T, action_dim, device=self.device)
        
        # 获取状态特征 (共享主干的输出)
        with torch.no_grad():
            state_features = self._extract_backbone_features(data)
        
        all_steps = [] if return_all_steps else None
        
        # 逐步去噪
        for t_step in reversed(range(self.num_diffusion_steps)):
            t = torch.full((B,), t_step, dtype=torch.long, device=self.device)
            x_t, pred_x0 = self.p_sample_step(x_t, t, state_features)
            
            if return_all_steps:
                all_steps.append(pred_x0.clone())
        
        # 返回最终动作 (可以选择均值或采样)
        action = x_t  # 或 pred_x0
        
        if return_all_steps:
            return action, all_steps
        return action
    
    def _extract_backbone_features(self, data):
        """
        提取共享主干的特征 (用于去噪过程)
        """
        # 编码图像
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
        
        # 添加额外模态
        encoded.append(self.extra_encoder(data["obs"]))
        encoded = torch.cat(encoded, -1)
        
        # 语言编码
        lang_h = self.language_encoder(data)
        encoded = torch.cat(
            [encoded, lang_h.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1
        )
        
        # RNN编码
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
        output, _ = self.rnn(encoded, (h0, c0))
        
        return output  # (B, T, feature_dim)
    
    def get_denoising_target(self, data, actions, t):
        """
        获取在时间步t的去噪目标 (用于训练专家)
        
        Args:
            data: 输入数据
            actions: (B, T, action_dim) - 真实动作
            t: (B,) - 时间步
        Returns:
            target_action: (B, T, action_dim) - 去噪目标动作
        """
        # 添加噪声
        noise = torch.randn_like(actions)
        x_t = self.q_sample(actions, t, noise)
        
        # 预测去噪后的动作 (简化: 直接使用真实动作作为目标)
        # 实际应该使用训练好的去噪网络预测
        target_action = actions  # 简化实现
        
        return target_action, x_t
