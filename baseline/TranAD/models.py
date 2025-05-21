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



# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
	def __init__(self, feats):
		super(TranAD_Basic, self).__init__()
		self.name = 'TranAD_Basic'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = CompatibleTransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = CompatibleTransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		# Ensure correct dimensions: (seq_len, batch_size, features)
		src = src.permute(1, 0, 2) if src.size(0) == 1 else src
		memory = self.transformer_encoder(src, src_key_padding_mask=None)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD_Transformer(nn.Module):
	def __init__(self, feats):
		super(TranAD_Transformer, self).__init__()
		self.name = 'TranAD_Transformer'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_hidden = 6
		self.n_window = 10
		self.n = 2 * self.n_feats * self.n_window
		self.transformer_encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
		self.transformer_decoder1 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.transformer_decoder2 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src.permute(1, 0, 2).flatten(start_dim=1)
		tgt = self.transformer_encoder(src)
		return tgt

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.transformer_decoder1(self.encode(src, c, tgt))
		x1 = x1.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x1 = self.fcn(x1)
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.transformer_decoder2(self.encode(src, c, tgt))
		x2 = x2.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x2 = self.fcn(x2)
		return x1, x2

# Proposed Model + Self Conditioning + MAML (VLDB 22)
class TranAD_Adversarial(nn.Module):
	def __init__(self, feats):
		super(TranAD_Adversarial, self).__init__()
		self.name = 'TranAD_Adversarial'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = CompatibleTransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = CompatibleTransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode_decode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		# Ensure correct dimensions: (seq_len, batch_size, features)
		src = src.permute(1, 0, 2) if src.size(0) == 1 else src
		memory = self.transformer_encoder(src, src_key_padding_mask=None)
		tgt = tgt.repeat(1, 1, 2)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x = self.encode_decode(src, c, tgt)
		# Phase 2 - With anomaly scores
		c = (x - src) ** 2
		x = self.encode_decode(src, c, tgt)
		return x

# Proposed Model + Adversarial + MAML (VLDB 22)
class TranAD_SelfConditioning(nn.Module):
	def __init__(self, feats):
		super(TranAD_SelfConditioning, self).__init__()
		self.name = 'TranAD_SelfConditioning'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = CompatibleTransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = CompatibleTransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = CompatibleTransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		# Ensure correct dimensions: (seq_len, batch_size, features)
		src = src.permute(1, 0, 2) if src.size(0) == 1 else src
		memory = self.transformer_encoder(src, src_key_padding_mask=None)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
	def __init__(self, feats):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = CompatibleTransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = CompatibleTransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = CompatibleTransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		# Ensure correct dimensions: (seq_len, batch_size, features)
		src = src.permute(1, 0, 2) if src.size(0) == 1 else src
		memory = self.transformer_encoder(src, mask=None, src_key_padding_mask=None)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2
