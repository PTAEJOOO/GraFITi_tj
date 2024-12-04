import argparse
import sys
import time
from random import SystemRandom

from tqdm import tqdm

# fmt: off
parser = argparse.ArgumentParser(description="Training Script for USHCN dataset.")
parser.add_argument("-q", "--quiet", default=False, const=True, help="kernel-inititialization", nargs="?")
parser.add_argument("-r", "--run_id", default=None, type=str, help="run_id")
parser.add_argument("-c", "--config", default=None, type=str, help="load external config", nargs=2)
parser.add_argument("-e", "--epochs", default=300, type=int, help="maximum epochs")
parser.add_argument("-f", "--fold", default=2, type=int, help="fold number")
parser.add_argument("-bs", "--batch-size", default=32, type=int, help="batch-size")
parser.add_argument("-lr", "--learn-rate", default=0.001, type=float, help="learn-rate")
parser.add_argument("-b", "--betas", default=(0.9, 0.999), type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001, type=float, help="weight-decay")
parser.add_argument("-hs", "--hidden-size", default=32, type=int, help="hidden-size")
parser.add_argument("-ki", "--kernel-init", default="skew-symmetric", help="kernel-inititialization")
parser.add_argument("-n", "--note", default="", type=str, help="Note that can be added")
parser.add_argument("-s", "--seed", default=None, type=int, help="Set the random seed.")
parser.add_argument("-nl", "--nlayers", default=2, type=int, help="")
parser.add_argument("-ahd", "--attn-head", default=2, type=int, help="")
parser.add_argument("-ldim", "--latent-dim", default=128, type=int, help="")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-ft", "--forc-time", default=0, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond-time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("-ax", "--auxiliary", default=False, const=True, help="use auxiliary node", nargs="?")
parser.add_argument("-wocat", "--wocat", default=False, const=True, help="with-out channel attention", nargs="?")

# fmt: on

ARGS = parser.parse_args()

import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

import logging
import os
import random
import warnings

import numpy as np
import torch
import torchinfo
from IPython.core.display import HTML
from torch import Tensor, jit
import pdb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
logging.basicConfig(level=logging.WARN)
HTML("<style>.jp-OutputArea-prompt:empty {padding: 0; border: 0;}</style>")

if not os.path.exists('saved_models/'):
    os.makedirs('saved_models/')

if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": ARGS.betas,
    # "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

if ARGS.dataset == "ushcn":
    from tsdm.tasks import USHCN_DeBrouwer2019

    TASK = USHCN_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon=ARGS.forc_time,
                               num_folds=ARGS.nfolds)
elif ARGS.dataset == "mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

    TASK = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon=ARGS.forc_time,
                                   num_folds=ARGS.nfolds)
elif ARGS.dataset == "mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

    TASK = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon=ARGS.forc_time,
                              num_folds=ARGS.nfolds)
elif ARGS.dataset == 'physionet2012':
    from tsdm.tasks.physionet2012 import Physionet2012

    TASK = Physionet2012(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon=ARGS.forc_time,
                         num_folds=ARGS.nfolds)

from gratif.gratif import tsdm_collate

dloader_config_test = {
    "batch_size": 1798,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": tsdm_collate,
}

TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_test)

################################################################################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pypots.imputation import SAITS
# from pypots.utils.metrics import calc_mae
# from pygrinder import mcar
from utils import *
from torch.utils.data import DataLoader

for batch in tqdm(TEST_LOADER):
    x_time, x_vals, x_mask, y_time, y_vals, y_mask = (tensor.to(DEVICE) for tensor in batch)
    obs_data = x_vals[:,:37,:]
    dataset = {"X": obs_data}  # X for model input

    load_saits = SAITS(n_steps=37, n_features=37, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1)
    load_saits.load("saved_imputer/saits_physionet2012_ep100.pypots")  # reload the serialized model file for following imputation or training

    imputation = load_saits.impute(dataset)
    print("Impute test dataset successfully!")

    x_mask_ = torch.ones_like(x_mask[:,:37,:]).to(DEVICE)
    x_mask = torch.cat([x_mask_, x_mask[:,37:,:]],dim=1)
    new_x = torch.cat([torch.from_numpy(imputation).to(DEVICE), x_vals[:,37:,:]],dim=1)

    new_dataset = CustomDataset(x_time,new_x,x_mask,y_time,y_vals,y_mask)
    NEW_TEST_LOADER = DataLoader(new_dataset, batch_size=ARGS.batch_size, shuffle=True)
    print("NEW TEST LOADER!")

################################################################################


def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.mean((y[mask] - yhat[mask]) ** 2)
    return err


def MAE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sum(mask * torch.abs(y - yhat), 1) / (torch.sum(mask, 1))
    return torch.mean(err)


def RMSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
    err = torch.sqrt(torch.sum(mask * (y - yhat) ** 2, 1) / (torch.sum(mask, 1)))
    return torch.mean(err)


METRICS = {
    "RMSE": jit.script(RMSE),
    "MSE": jit.script(MSE),
    "MAE": jit.script(MAE),
}
LOSS = jit.script(MSE)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from gratif.gratif import GrATiF

MODEL_CONFIG = {
    "input_dim": TASK.dataset.shape[-1],
    "attn_head": ARGS.attn_head,
    "latent_dim": ARGS.latent_dim,
    "n_layers": ARGS.nlayers,
    "device": DEVICE,
    "auxiliary": ARGS.auxiliary,
    "wocat": ARGS.wocat,
    "cond_time": ARGS.cond_time
}

MODEL = GrATiF(**MODEL_CONFIG).to(DEVICE)

def predict_fn(model, batch) -> tuple[Tensor, Tensor, Tensor]:
    """Get targets and predictions."""
    T, X, M, TY, Y, MY = (tensor.to(DEVICE) for tensor in batch)
    output, target_U_, target_mask_ = model(T, X, M, TY, Y, MY)
    return target_U_, output.squeeze(), target_mask_

MODEL.zero_grad(set_to_none=True)

if ARGS.auxiliary:
    if ARGS.wocat:
        chp = torch.load('saved_grafiti_impute/' + ARGS.dataset + '_' + str(ARGS.nlayers) + '_' + str(ARGS.attn_head) + '_' + str(ARGS.latent_dim) + '_wo_ax' + '.h5', map_location=torch.device('cpu'))
    else:
        chp = torch.load('saved_grafiti_impute/' + ARGS.dataset + '_' + str(ARGS.nlayers) + '_' + str(ARGS.attn_head) + '_' + str(ARGS.latent_dim) + '_ax' + '.h5', map_location=torch.device('cpu'))
else:
    if ARGS.wocat:
        chp = torch.load('saved_grafiti_impute/' + ARGS.dataset + '_' + str(ARGS.nlayers) + '_' + str(ARGS.attn_head) + '_' + str(ARGS.latent_dim) + '_wo_og.h5', map_location=torch.device('cpu'))
    else:
        chp = torch.load('saved_grafiti_impute/' + ARGS.dataset + '_' + str(ARGS.nlayers) + '_' + str(ARGS.attn_head) + '_' + str(ARGS.latent_dim) + '.h5', map_location=torch.device('cpu'))
MODEL.load_state_dict(chp['state_dict'])
loss_list = []
count = 0
with torch.no_grad():
    for batch in tqdm(NEW_TEST_LOADER):
        # Forward
        Y, YHAT, MASK = predict_fn(MODEL, batch)
        R = LOSS(Y, YHAT, MASK)
        assert torch.isfinite(R).item(), "Model Collapsed!"
        # loss_list.append([R*Y.shape[0]])
        loss_list.append([R * MASK.sum()])
        count += MASK.sum()
test_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
print("test_loss : ", test_loss.item())

with open("log/eval_grafiti_impute.txt", "a") as file:
    if ARGS.auxiliary:
        if ARGS.wocat:
            content = ARGS.dataset + '_' + str(ARGS.nlayers) + '_' + str(ARGS.attn_head) + '_' + str(ARGS.latent_dim) + '_wo_ax' + ' : ' + str(test_loss.item())
        else:
            content = ARGS.dataset + '_' + str(ARGS.nlayers) + '_' + str(ARGS.attn_head) + '_' + str(ARGS.latent_dim) + '_ax' + ' : ' + str(test_loss.item())
    else:
        if ARGS.wocat:
            content = ARGS.dataset + '_' + str(ARGS.nlayers) + '_' + str(ARGS.attn_head) + '_' + str(ARGS.latent_dim) + '_wo_og : ' + str(test_loss.item())
        else:
            content = ARGS.dataset + '_' + str(ARGS.nlayers) + '_' + str(ARGS.attn_head) + '_' + str(ARGS.latent_dim) + ' : ' + str(test_loss.item())
    file.write(content + "\n")