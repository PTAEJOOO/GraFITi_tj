import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from gratif import gratif_layers
from torch.nn.utils.rnn import pad_sequence
import pdb

import math
import lib.utils as utils
from lib.evaluation import *


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor


def tsdm_collate(batch: list[Sample]) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        sorted_idx = torch.argsort(t)

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

        x_vals.append(x[sorted_idx])
        x_time.append(t[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)

        context_x.append(torch.cat([t, t_target], dim=0))
        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True).squeeze(),
        x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0).squeeze(),
        x_mask=pad_sequence(context_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(context_x, batch_first=True).squeeze(),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0).squeeze(),
        y_mask=pad_sequence(target_mask, batch_first=True).squeeze(),
    )

###########################################################################################################################
class nconv(nn.Module):
	def __init__(self):
		super(nconv,self).__init__()

	def forward(self, x, A):
		# x (B, F, N, M)
		# A (B, M, N, N)
		x = torch.einsum('bfnm,bmnv->bfvm',(x,A)) # used
		# print(x.shape)
		return x.contiguous() # (B, F, N, M)

class linear(nn.Module):
	def __init__(self, c_in, c_out):
		super(linear,self).__init__()
		# self.mlp = nn.Linear(c_in, c_out)
		self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

	def forward(self, x):
		# x (B, F, N, M)

		# return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
		return self.mlp(x)
		
class gcn(nn.Module):
	def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
		super(gcn,self).__init__()
		self.nconv = nconv()
		c_in = (order*support_len+1)*c_in
		# c_in = (order*support_len)*c_in
		self.mlp = linear(c_in, c_out)
		self.dropout = dropout
		self.order = order

	def forward(self, x, support):
		# x (B, F, N, M)
		# a (B, M, N, N)
		out = [x]
		for a in support:
			x1 = self.nconv(x,a)
			out.append(x1)
			for k in range(2, self.order + 1):
				x2 = self.nconv(x1,a)
				out.append(x2)
				x1 = x2

		h = torch.cat(out, dim=1) # concat x and x_conv
		h = self.mlp(h)
		return F.relu(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        """
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    
###########################################################################################################################

class GrATiF_patch(nn.Module):

    def __init__(
            self,
            args,
            input_dim=41,
            attn_head=4,
            latent_dim=128,
            n_layers=2,
            device='cuda'):
        super().__init__()
        self.dim = input_dim
        self.attn_head = attn_head
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.device = device
        self.enc = gratif_layers.Encoder(self.dim, self.latent_dim, self.n_layers, self.attn_head, device=device)

        ### Intra-time series modeling ## 
		## Time embedding\
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim-1)

    def LearnableTE(self, tt):
		# tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))

        return torch.cat([out1, out2], -1)
    
    def TTCN(self, X_int, mask_X):
		# X_int: shape (B*N*M, L, F)
		# mask_X: shape (B*N*M, L, 1)
        
        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int) # (N, Lx, F_in*ttcn_dim)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
		# normalize along with sequence dimension
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*ttcn_dim)
        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1) # (N, Lx, ttcn_dim, F_in)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (N, ttcn_dim)
        h_t = torch.relu(ttcn_out + self.T_bias) # (N, ttcn_dim)
        
        return h_t

    def get_extrapolation(self, context_x, context_w, target_x, target_y):
        context_mask = context_w[:, :, self.dim:]
        X = context_w[:, :, :self.dim]
        X = X * context_mask
        context_mask = context_mask + target_y[:, :, self.dim:]
        # self.enc 가기 전에 dimension을 맞춰줘야함..
        output, target_U_, target_mask_ = self.enc(context_x, X, context_mask, target_y[:, :, :self.dim],
                                                   target_y[:, :, self.dim:])
        return output, target_U_, target_mask_

    def convert_data(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        return x_time, torch.cat([x_vals, x_mask], -1), y_time, torch.cat([y_vals, y_mask], -1)

    def forecasting(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
		
        """ 
		x_time (B, M, L, N) [0, 1]
        x_vals (B, M, L, N) 
		x_mask (B, M, L, N)
            
        y_time (B, L) [0, 1]
        y_vals (B, L_out, N) 
		y_mask (B, L_out, N)

		To ====>

        
        """
            
        ##
        B, M, L_in, N = x_vals.shape

        
        self.batch_size = B
        x_vals = x_vals.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
        x_time = x_time.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
        x_mask = x_mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
        te_his = self.LearnableTE(x_time) # (B*N*M, L, F_te) <=> embedded time points
        
        x_vals_ = torch.cat([x_vals, te_his], dim=-1)  # (B*N*M, L, F)

        # mask for the patch
        mask_patch = (x_mask.sum(dim=1) > 0) # (B*N*M, 1)

		### TTCN for patch modeling ###
        x_patch = self.TTCN(x_vals_, x_mask) # (B*N*M, hid_dim-1) <=> TTCN for patch embedding
        x_patch = torch.cat([x_patch, mask_patch],dim=-1) # (B*N*M, hid_dim) <=> incorporate a patch masking term into the patch embedding
        x_patch = x_patch.view(self.batch_size, self.N, self.M, -1) # (B, N, M, hid_dim)
        B, N, M, D = x_patch.shape
        ##
        
        ## x_time: (B,M), x_vals & x_maks: (B,M,N) 으로 맞춰줘야 할듯
        context_x, context_y, target_x, target_y = self.convert_data(x_time, x_vals, x_mask, y_time, y_vals, y_mask)
        # pdb.set_trace()
        if len(context_y.shape) == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            target_x = target_x.unsqueeze(0)
            target_y = target_y.unsqueeze(0)
        output, target_U_, target_mask_ = self.get_extrapolation(context_x, context_y, target_x, target_y)

        return output, target_U_, target_mask_.to(torch.bool)