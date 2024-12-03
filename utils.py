import torch
from torch.utils.data import DataLoader, Dataset
from typing import NamedTuple
from torch import Tensor


class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.


class CustomDataset(Dataset):
    def __init__(self, T, X, M, TY, Y, MY):
        self.T = T
        self.X = X
        self.M = M
        self.TY = TY
        self.Y = Y
        self.MY = MY

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, idx):
        return Batch(
            x_time=self.T[idx],
            x_vals=self.X[idx],
            x_mask=self.M[idx],
            y_time=self.TY[idx],
            y_vals=self.Y[idx],
            y_mask=self.MY[idx],
        )