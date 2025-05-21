import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn import TransformerEncoderLayer as BaseTransformerEncoderLayer
from torch.nn import TransformerDecoderLayer as BaseTransformerDecoderLayer
from src.dlutils import *

class CompatibleTransformerEncoderLayer(BaseTransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        return super().forward(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)

class CompatibleTransformerDecoderLayer(BaseTransformerDecoderLayer):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=None):
        return super().forward(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
from constants import *
torch.manual_seed(1)



## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(feats, self.n_feats)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
			out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)
