import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from gratif import gratif_layers
from torch.nn.utils.rnn import pad_sequence
import pdb


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
            wocat=False):
        super().__init__()
        self.auxiliary = auxiliary
        self.wocat = wocat
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
        # print(context_x.size())
        # print(context_mask.size())
        # print(target_y.size())
        # print(target_y[:, :, :self.dim].size())
        # print(target_y[:, :, self.dim:].size())
        # print(target_x)
        # print(target_y[0,10:15,:self.dim])
        # print(target_y[0,10:15,self.dim:])
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

    def forward(self, x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        # print(x_time.size(), x_vals.size(), x_mask.size())
        # print(y_time.size(), y_vals.size(), y_mask.size())
        # print(torch.sum(y_vals[0], 0)) => each timepoint를 기준으로 all channels의 observation value를 더한 값, if 0 => 그 timepoint에 all channels에 관측된 게 없음.
        # print(torch.sum(y_mask[0], 0))
        # print(torch.sum(y_vals[0], 1)) => each channel을 기준으로 all timepoints에 observation value를 더한 값, if 0 => 그 channel의 all timepoints에 관측 된게 없음
        # print(torch.sum(y_mask[0], 1))
        if self.auxiliary:
            x_time, x_vals, x_mask, y_time, y_vals, y_mask = self.add_auxiliary(x_time, x_vals, x_mask, y_time, y_vals, y_mask)

        context_x, context_y, target_x, target_y = self.convert_data(x_time, x_vals, x_mask, y_time, y_vals, y_mask)
        # print(context_x.size(), target_x.size())
        # print(context_y.size(), target_y.size())
        # pdb.set_trace()
        if len(context_y.shape) == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            target_x = target_x.unsqueeze(0)
            target_y = target_y.unsqueeze(0)
        output, target_U_, target_mask_ = self.get_extrapolation(context_x, context_y, target_x, target_y)

        return output, target_U_, target_mask_.to(torch.bool)