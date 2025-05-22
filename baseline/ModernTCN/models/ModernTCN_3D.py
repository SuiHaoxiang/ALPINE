import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ModernTCN_Layer import series_decomp, Flatten_Head
from layers.RevIN import RevIN

class Stage_3D(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, 
                 dmodel, dw_model, nvars, small_kernel_merged=False, drop=0.1):
        super(Stage_3D, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Conv1d(dmodel, dmodel, kernel_size=large_size,
                        padding=large_size//2, groups=dmodel//dw_model),
                nn.GELU(),
                nn.Conv1d(dmodel, dmodel, kernel_size=small_size,
                        padding=small_size//2, groups=dmodel//dw_model),
                nn.Dropout(drop),
                nn.Conv1d(dmodel, dmodel, kernel_size=1),
            )
            self.blocks.append(block)

        self.ffn = nn.Sequential(
            nn.Conv1d(dmodel, dmodel * ffn_ratio, kernel_size=1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(dmodel * ffn_ratio, dmodel, kernel_size=1),
            nn.Dropout(drop)
        )
        self.norm = nn.LayerNorm(dmodel)

    def forward(self, x):
        B, D, L = x.shape
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual

        residual = x
        x = self.ffn(x)
        x = x + residual

        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x

class ModernTCN_3D(nn.Module):
    def __init__(self, configs):
        super(ModernTCN_3D, self).__init__()
        self.task_name = configs.task_name
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims
        # 动态设置输入通道数
        if hasattr(configs, 'actual_enc_in'):
            self.nvars = configs.actual_enc_in
        else:
            self.nvars = configs.enc_in
        self.revin = configs.revin
        self.dropout = configs.dropout
        
        if self.revin:
            self.revin_layer = RevIN(self.nvars, affine=configs.affine, 
                                   subtract_last=configs.subtract_last)

        # 动态设置stem层维度
        self.stem = nn.Conv1d(
            in_channels=self.nvars,  # 使用实际输入通道数
            out_channels=self.dims[0], 
            kernel_size=1,
            bias=False
        )
        print(f"Initialized stem layer with in_channels={self.nvars}, out_channels={self.dims[0]}")
        
        self.stages = nn.ModuleList()
        for stage_idx in range(len(configs.num_blocks)):
            stage = Stage_3D(
                configs.ffn_ratio,
                configs.num_blocks[stage_idx],
                configs.large_size[stage_idx],
                configs.small_size[stage_idx],
                dmodel=self.dims[stage_idx],
                dw_model=self.dw_dims[stage_idx],
                nvars=self.nvars,
                drop=self.dropout
            )
            self.stages.append(stage)

        if self.task_name == 'anomaly_detection':
            self.head = nn.Linear(self.dims[-1], 1)

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
            
        x = x.permute(0, 2, 1)  # [B,L,D] -> [B,D,L]
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
            
        if self.task_name == 'anomaly_detection':
            x = self.head(x.permute(0, 2, 1))  # [B,D,L] -> [B,L,1]
            if self.revin:
                x = self.revin_layer(x, 'denorm')
            return x
            
        return x
