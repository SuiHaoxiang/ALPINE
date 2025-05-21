# # 修复后的 ModernTCN 模块相关代码
import torch
from torch import nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN
from models.ModernTCN_Layer import series_decomp, Flatten_Head
from typing import Tuple

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, 
                 dmodel, dw_model, nvars, small_kernel_merged=False, drop=0.1):
        super(Stage, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Conv1d(dmodel, dmodel, kernel_size=large_size,
                          padding=large_size//2, groups=dmodel),
                nn.GELU(),
                nn.Conv1d(dmodel, dmodel, kernel_size=small_size,
                          padding=small_size//2, groups=dmodel),
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

        # LayerNorm in [B,D,L] -> [B,L,D] -> norm -> [B,D,L]
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = ModernTCN(configs)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x.permute(0, 2, 1)
        return self.model(x)

class ModernTCN(nn.Module):
    def __init__(self, configs):
        super(ModernTCN, self).__init__()
        self.task_name = configs.task_name
        self.patch_size = 25  # 修改为25以保持更大维度
        self.patch_stride = 5  # 修改为5以减少下采样
        self.stem_ratio = getattr(configs, 'stem_ratio', 6)
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = getattr(configs, 'ffn_ratio', 1)
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims
        self.seq_len = configs.seq_len
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last
        self.nvars = configs.enc_in
        self.small_kernel_merged = getattr(configs, 'small_kernel_merged', False)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.use_multi_scale = getattr(configs, 'use_multi_scale', False)

        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=self.affine, subtract_last=self.subtract_last)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Linear(1, self.dims[0])
        self.downsample_layers.append(stem)

        self.num_stage = len(self.num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(self.dims[i]),
                    nn.Conv1d(self.dims[i], self.dims[i + 1],
                              kernel_size=self.downsample_ratio,
                              stride=self.downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(
                self.ffn_ratio, 
                self.num_blocks[stage_idx], 
                self.large_size[stage_idx], 
                self.small_size[stage_idx],
                dmodel=self.dims[stage_idx],
                dw_model=self.dw_dims[stage_idx],
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                drop=self.dropout
            )
            self.stages.append(layer)

        if self.task_name == 'anomaly_detection':
            self.head_dection1 = nn.Linear(self.dims[-1], self.seq_len)  # 改为seq_len以匹配输入维度

    def forward_feature(self, x, te=None):
        B, _, D, L = x.shape
        if D != 1:
            x = x.permute(0, 1, 3, 2)
            B, _, D, L = x.shape

        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, 1)
        x = self.downsample_layers[0](x)
        x = x.reshape(B, L, -1)

        for i in range(self.num_stage):
            if i > 0:
                x = x.permute(0, 2, 1)
                x = self.downsample_layers[i](x)
                x = x.permute(0, 2, 1)
            x = x.permute(0, 2, 1)
            x = self.stages[i](x)
            x = x.permute(0, 2, 1)
        return x

    def detection(self, x):
        pass

    def forward(self, x, te=None):
        if self.revin:
            x = self.revin_layer(x, 'norm')

        B, L, D = x.shape
        x = x.permute(0, 2, 1)
        x = self.forward_feature(x.unsqueeze(1), te)

        if self.task_name == 'anomaly_detection':
            x = self.head_dection1(x.squeeze(1))
            x = x.permute(0, 2, 1)
            if self.revin:
                x = self.revin_layer(x, 'denorm')
            return x

        return None

    def structural_reparam(self):
        pass
