"""
LoRA (Low-Rank Adaptation) Adapter for Expert Policies
实现参数高效的专家适配头
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    """
    LoRA适配器: 通过低秩矩阵分解实现参数高效的适配
    
    对于输入特征 z, 输出动作计算为:
    a_e = W_freeze * z + ΔW_e * z + (b_freeze + Δb_e)
    
    其中 ΔW_e = U_e * V_e^T (低秩分解, rank r << dim(z))
    """
    
    def __init__(
        self,
        input_dim,
        output_dim,
        rank=10,
        alpha=1.0,
        freeze_weight=None,
        freeze_bias=None,
    ):
        """
        Args:
            input_dim: 输入特征维度 (共享主干的输出维度)
            output_dim: 输出动作维度
            rank: LoRA的秩 (r << input_dim)
            alpha: LoRA的缩放因子
            freeze_weight: 冻结的权重矩阵 W_freeze (可选)
            freeze_bias: 冻结的偏置 b_freeze (可选)
        """
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        # 缓存缩放项，避免在每次前向时重复做除法
        self.scaling = alpha / rank
        
        # 低秩矩阵分解: ΔW = U * V^T
        # U: (output_dim, rank), V: (input_dim, rank)
        self.lora_U = nn.Parameter(torch.zeros(output_dim, rank))
        self.lora_V = nn.Parameter(torch.zeros(input_dim, rank))
        
        # 可学习的偏置增量
        self.lora_bias = nn.Parameter(torch.zeros(output_dim))
        
        # 冻结的权重和偏置 (来自教师策略)
        self.register_buffer('freeze_weight', freeze_weight)
        self.register_buffer('freeze_bias', freeze_bias)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_U, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_V)
        nn.init.zeros_(self.lora_bias)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) 或 (B, input_dim) - 共享主干的输出特征
        Returns:
            output: (B, T, output_dim) 或 (B, output_dim) - 专家动作
        """
        # 计算低秩增量: ΔW * x = U * (V^T * x)
        # torch.matmul 可同时处理 2D/3D 输入，因此这里统一计算路径
        lora_hidden = torch.matmul(x, self.lora_V)        # (..., rank)
        lora_output = torch.matmul(lora_hidden, self.lora_U.t()) * self.scaling  # (..., output_dim)
        
        # 如果有冻结权重, 加上冻结部分
        if self.freeze_weight is not None:
            # F.linear 语义等价于 x @ W^T + b，且支持 2D/3D 的最后一维线性变换
            freeze_output = F.linear(x, self.freeze_weight, self.freeze_bias)
            output = freeze_output + lora_output + self.lora_bias
        else:
            output = lora_output + self.lora_bias
            
        return output
    
    def get_num_params(self):
        """返回LoRA适配器的参数量"""
        return (
            self.lora_U.numel() + 
            self.lora_V.numel() + 
            self.lora_bias.numel()
        )
