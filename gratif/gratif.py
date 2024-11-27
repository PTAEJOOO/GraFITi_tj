import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from gratif import gratif_layers
from torch.nn.utils.rnn import pad_sequence
import pdb

import sys


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


class GrATiF(nn.Module):

    def __init__(
            self,
            input_dim=41,
            attn_head=4,
            latent_dim=128,
            n_layers=2,
            device='cuda',
            auxiliary=False,
            wocat=False,
            imputer=None,
            cond_time=36):
        super().__init__()
        self.auxiliary = auxiliary
        self.wocat = wocat
        self.imputer = imputer
        self.cond_time = cond_time
        if auxiliary:
            self.dim = input_dim + 1
        else:
            self.dim = input_dim
        self.attn_head = attn_head
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.device = device
        self.enc = gratif_layers.Encoder(self.dim, self.latent_dim, self.n_layers, self.attn_head, device=device, wocat=wocat)

    def get_extrapolation(self, context_x, context_w, target_x, target_y):
        context_mask = context_w[:, :, self.dim:] # context_w[:, :, self.dim:] = original x_mask
        X = context_w[:, :, :self.dim] # context_w[:, :, :self.dim] = original x_vals
        X = X * context_mask
        context_mask = context_mask + target_y[:, :, self.dim:] # observation mask || target mask
        output, target_U_, target_mask_ = self.enc(context_x, X, context_mask, target_y[:, :, :self.dim],
                                                   target_y[:, :, self.dim:])
        return output, target_U_, target_mask_

    def convert_data(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        return x_time, torch.cat([x_vals, x_mask], -1), y_time, torch.cat([y_vals, y_mask], -1)
    
    def add_auxiliary(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        bs, L, C = x_vals.shape

        x_time = torch.concat((x_time, torch.zeros(bs,1).to(self.device)), dim=-1) # torch.ones(bs,1)
        x_vals = torch.concat((x_vals, torch.zeros(bs,1,C).to(self.device)), dim=1)
        x_vals = torch.concat((x_vals, torch.zeros(bs,L+1,1).to(self.device)), dim=-1)
        x_mask = torch.concat((x_mask, torch.ones(bs,1,C).to(self.device)), dim=1)
        new_c = (x_mask.sum(dim=-1) > 0).int().to(self.device)
        x_mask = torch.concat((x_mask, new_c.unsqueeze(-1)), dim=-1)

        y_time = torch.concat((y_time, torch.zeros(bs,1).to(self.device)), dim=-1) # torch.ones(bs,1)
        y_vals = torch.concat((y_vals, torch.zeros(bs,1,C).to(self.device)), dim=1)
        y_vals = torch.concat((y_vals, torch.zeros(bs,L+1,1).to(self.device)), dim=-1)
        y_mask = torch.concat((y_mask, torch.zeros(bs,1,C).to(self.device)), dim=1)
        y_mask = torch.concat((y_mask, torch.zeros(bs,L+1,1).to(self.device)), dim=-1)

        return x_time, x_vals, x_mask, y_time, y_vals, y_mask

    def impute(self, x_vals, x_mask):
        # x_vals on only observation range for imputation
        temp_x_vals = x_vals[:,self.cond_time:,:]
        temp_x_mask = x_mask[:,self.cond_time:,:]
        x_vals_ = x_vals[:,:self.cond_time,:]
        x_mask_ = x_mask[:,:self.cond_time,:]

        B,T,C = x_vals_.shape
        x_vals_ = x_vals_.view(B,T*C).transpose(0,1)
        x_mask_ = x_mask_.view(B,T*C).transpose(0,1)
        # impute
        x_vals_ = self.imputer.predict(x_vals_, x_mask_) # to-do: load pretrained imputer and transform the x_vals
        x_vals_ = torch.from_numpy(x_vals_).transpose(0,1).reshape(B,T,C)
        x_mask_ = torch.ones_like(x_mask[:,:self.cond_time,:])
        # concat
        x_vals_ = torch.concat((x_vals_, temp_x_vals), dim=1)
        x_mask_ = torch.concat((x_mask_, temp_x_mask), dim=1)
        return x_vals_, x_mask_

    def forward(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        print(f"x_vals shape = {x_vals.shape}")
        print(x_mask[0].sum(dim=-1))
        print(y_mask[0].sum(dim=-1))
        print(x_mask[1].sum(dim=-1))
        print(y_mask[1].sum(dim=-1))
        print(x_mask[2].sum(dim=-1))
        print(y_mask[2].sum(dim=-1))
        sys.exit(0)
        if self.auxiliary:
            x_time, x_vals, x_mask, y_time, y_vals, y_mask = self.add_auxiliary(x_time, x_vals, x_mask, y_time, y_vals, y_mask)

        if self.imputer is not None:
            x_vals, x_mask = self.impute(x_vals, x_mask)

        context_x, context_y, target_x, target_y = self.convert_data(x_time, x_vals, x_mask, y_time, y_vals, y_mask)
        # pdb.set_trace()
        if len(context_y.shape) == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            target_x = target_x.unsqueeze(0)
            target_y = target_y.unsqueeze(0)
        output, target_U_, target_mask_ = self.get_extrapolation(context_x, context_y, target_x, target_y)

        return output, target_U_, target_mask_.to(torch.bool)