import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.orig_len = x.size(-1)  # 保存原始序列长度
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        
        # 处理序列长度变化
        current_len = x.size(-1)
        if current_len != self.orig_len:
            # 计算缩放因子
            scale_factor = current_len / self.orig_len
            # 调整统计量维度
            stdev = self.stdev.repeat(1, 1, int(scale_factor * self.stdev.size(-1)))[:,:,:current_len]
            if self.subtract_last:
                last = self.last.repeat(1, 1, int(scale_factor * self.last.size(-1)))[:,:,:current_len]
            else:
                mean = self.mean.repeat(1, 1, int(scale_factor * self.mean.size(-1)))[:,:,:current_len]
        else:
            stdev = self.stdev
            if self.subtract_last:
                last = self.last
            else:
                mean = self.mean
        
        x = x * stdev
        if self.subtract_last:
            x = x + last
        else:
            x = x + mean
        
        return x
