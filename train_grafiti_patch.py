import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim

import lib.utils as utils
from lib.parse_datasets import parse_datasets
from tPatchGNN.model.tPatchGNN import *

from random import SystemRandom

from tqdm import tqdm

import logging
import os
import random
import warnings

import torchinfo
from IPython.core.display import HTML
from torch import Tensor, jit
import pdb

#####################################################################################################
## from tpatchgnn
parser = argparse.ArgumentParser('IMTS Forecasting')

parser.add_argument('--state', type=str, default='def')
parser.add_argument('-n',  type=int, default=int(1e8), help="Size of the dataset")
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=1, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")
# parser.add_argument('--epoch', type=int, default=1000, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
# parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
# parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
# parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
# parser.add_argument('--seed', type=int, default=1, help="Random seed")
# parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")

# value 0 means using original time granularity, Value 1 means quantization by 1 hour, 
# value 0.1 means quantization by 0.1 hour = 6 min, value 0.016 means quantization by 0.016 hour = 1 min
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Model name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

#####################################################################################################
## from grafiti
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

#####################################################################################################

ARGS = parser.parse_args()
print(' '.join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)

import yaml

if ARGS.config is not None:
    cfg_file, cfg_id = ARGS.config
    with open(cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)
        vars(ARGS).update(**cfg_dict[int(cfg_id)])

print(ARGS)

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

#####################################################################################################
ARGS.npatch = int(np.ceil((ARGS.history - ARGS.patch_size) / ARGS.stride)) + 1 # (window size for a patch)

os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu
file_name = os.path.basename(__file__)[:-3]
ARGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARGS.PID = os.getpid()
print("PID, device:", ARGS.PID, ARGS.device)

#####################################################################################################

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": ARGS.betas,
    # "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

#####################################################################################################
data_obj = parse_datasets(ARGS, patch_ts=True)
input_dim = data_obj["input_dim"]
# print(f"input dim = {input_dim}") # =41
### Model setting ###
ARGS.ndim = input_dim

#####################################################################################################

# if ARGS.dataset == "ushcn":
#     from tsdm.tasks import USHCN_DeBrouwer2019

#     TASK = USHCN_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon=ARGS.forc_time,
#                                num_folds=ARGS.nfolds)
# elif ARGS.dataset == "mimiciii":
#     from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

#     TASK = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon=ARGS.forc_time,
#                                    num_folds=ARGS.nfolds)
# elif ARGS.dataset == "mimiciv":
#     from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

#     TASK = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon=ARGS.forc_time,
#                               num_folds=ARGS.nfolds)
# elif ARGS.dataset == 'physionet2012':
#     from tsdm.tasks.physionet2012 import Physionet2012

#     TASK = Physionet2012(normalize_time=True, condition_time=ARGS.cond_time, forecast_horizon=ARGS.forc_time,
#                          num_folds=ARGS.nfolds)

from gratif.gratif import tsdm_collate

# dloader_config_train = {
#     "batch_size": ARGS.batch_size,
#     "shuffle": True,
#     "drop_last": True,
#     "pin_memory": True,
#     "num_workers": 4,
#     "collate_fn": tsdm_collate,
# }

# dloader_config_infer = {
#     "batch_size": 32,
#     "shuffle": False,
#     "drop_last": False,
#     "pin_memory": True,
#     "num_workers": 0,
#     "collate_fn": tsdm_collate,
# }

# TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
# INFER_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_infer)
# VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
# TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)
# EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}


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
    "device": DEVICE
}

MODEL = GrATiF(**MODEL_CONFIG).to(DEVICE)
torchinfo.summary(MODEL)


# def predict_fn(model, batch) -> tuple[Tensor, Tensor, Tensor]:
#     """Get targets and predictions."""
#     T, X, M, TY, Y, MY = (tensor.to(DEVICE) for tensor in batch)
#     output, target_U_, target_mask_ = model(T, X, M, TY, Y, MY)
#     return target_U_, output.squeeze(), target_mask_

num_batches = data_obj["n_train_batches"] # n_sample / batch_size
# batch = next(iter(TRAIN_LOADER))
MODEL.zero_grad(set_to_none=True)

from torch.optim import AdamW

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, 'min', patience=10, factor=0.5, min_lr=0.00001,
                                                       verbose=True)

es = False
best_val_loss = 10e8
total_num_batches = 0
for epoch in range(1, ARGS.epochs + 1):
    loss_list = []
    start_time = time.time()
    for batch in tqdm(num_batches):
        total_num_batches += 1
        OPTIMIZER.zero_grad()
        ###
        batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
        train_res = compute_all_losses(MODEL, batch_dict)
        train_res["loss"].backward()
        OPTIMIZER.step()
        ###
        # Y, YHAT, MASK = predict_fn(MODEL, batch)
        # R = LOSS(Y, YHAT, MASK)
        # assert torch.isfinite(R).item(), "Model Collapsed!"
        # loss_list.append([R])
        # # Backward
        # R.backward()
        # OPTIMIZER.step()
        sys.exit(0)
    # exit()    
    epoch_time = time.time()
    train_loss = torch.mean(torch.Tensor(loss_list))
    loss_list = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(VALID_LOADER):
            total_num_batches += 1
            # Forward
            Y, YHAT, MASK = predict_fn(MODEL, batch)
            R = LOSS(Y, YHAT, MASK)
            if R.isnan():
                pdb.set_trace()
            loss_list.append([R * MASK.sum()])
            count += MASK.sum()
    val_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
    print(epoch, "Train: ", train_loss.item(), " VAL: ", val_loss.item(), " epoch time: ", int(epoch_time - start_time),
          'secs')
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save({'args': ARGS,
                    'epoch': epoch,
                    'state_dict': MODEL.state_dict(),
                    'optimizer_state_dict': OPTIMIZER.state_dict(),
                    'loss': train_loss,
                    }, 'saved_models/' + ARGS.dataset + '_' + str(experiment_id) + '.h5')
        early_stop = 0
    else:
        early_stop += 1
    if early_stop == 30:
        print("Early stopping because of no improvement in val. metric for 30 epochs")
        es = True
    scheduler.step(val_loss)

    # LOGGER.log_epoch_end(epoch)
    if (epoch == ARGS.epochs) or (es == True):
        chp = torch.load('saved_models/' + ARGS.dataset + '_' + str(experiment_id) + '.h5')
        MODEL.load_state_dict(chp['state_dict'])
        loss_list = []
        count = 0
        with torch.no_grad():
            for batch in tqdm(TEST_LOADER):
                total_num_batches += 1
                # Forward
                Y, YHAT, MASK = predict_fn(MODEL, batch)
                R = LOSS(Y, YHAT, MASK)
                assert torch.isfinite(R).item(), "Model Collapsed!"
                # loss_list.append([R*Y.shape[0]])
                loss_list.append([R * MASK.sum()])
                count += MASK.sum()
        test_loss = torch.sum(torch.Tensor(loss_list).to(DEVICE) / count)
        print("Best_val_loss: ", best_val_loss.item(), " test_loss : ", test_loss.item())
        break
