import argparse
import sys
import time
from random import SystemRandom

from tqdm import tqdm

from imputers import *

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
parser.add_argument("-nf", "--nfolds", default=1, type=int, help="#folds for crossvalidation")
parser.add_argument("-ax", "--auxiliary", default=False, const=True, help="use auxiliary node", nargs="?")
parser.add_argument("-wocat", "--wocat", default=False, const=True, help="with-out channel attention", nargs="?")

parser.add_argument('--imputer', default="mice",  type=str, help="Name of the imputer: mean, knn, mf or mice")
parser.add_argument('--n-runs', type=int, default=5)
parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=True)
# SpatialKNNImputer params
parser.add_argument('--k', type=int, default=10)
# MFImputer params
parser.add_argument('--rank', type=int, default=10)
# MICEImputer params
parser.add_argument('--mice-iterations', type=int, default=10)
parser.add_argument('--mice-n-features', type=int, default=None)


# fmt: on

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

#############################################################################################
if ARGS.imputer == 'mean':
    imputer = MeanImputer(in_sample=ARGS.in_sample)
# elif ARGS.imputer == 'knn':
#     imputer = SpatialKNNImputer(adj=ARGS.adj, k=ARGS.k)
elif ARGS.imputer == 'mf':
    imputer = MatrixFactorizationImputer(rank=ARGS.rank)
elif ARGS.imputer == 'mice':
    imputer = MICEImputer(max_iter=ARGS.mice_iterations,
                              n_nearest_features=ARGS.mice_n_features,
                              in_sample=ARGS.in_sample)
elif ARGS.imputer == 'saits':
    imputer = MICEImputer(max_iter=ARGS.mice_iterations,
                              n_nearest_features=ARGS.mice_n_features,
                              in_sample=ARGS.in_sample)
#############################################################################################

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
from tsdm.tasks.physionet2012 import physionet_collate

dloader_config_train = {
    "batch_size": 100000,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 1,
    "collate_fn": tsdm_collate,
}

dloader_config_valid = {
    "batch_size": 100000,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": tsdm_collate,
}

dloader_config_test = {
    "batch_size": 100000,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
    "num_workers": 0,
    "collate_fn": tsdm_collate,
}

TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_valid)
TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_test)

######################################################################################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae
from pygrinder import mcar

for batch in tqdm(TRAIN_LOADER):
    x_time, x_vals, x_mask, y_time, y_vals, y_mask = (tensor.to(DEVICE) for tensor in batch)
    
    obs_data = x_vals[:,:37,:]
    X_ori = obs_data  # keep X_ori for validation
    X = mcar(obs_data, 0.3)  # randomly hold out 30% observed values as ground truth   
    dataset = {"X": X}  # X for model input

    saits = SAITS(n_steps=37, n_features=37, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1, epochs=200)
    saits.fit(dataset)
    imputation = saits.impute(dataset) # nan impute
    indicating_mask = np.isnan(X.cpu().numpy()) ^ np.isnan(X_ori.cpu().numpy())  # indicating mask for imputation error calculation
    mae = calc_mae(imputation, np.nan_to_num(X_ori.cpu().numpy()), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
    print(mae)
    saits.save("saved_imputer/saits_physionet2012_0.3_ep200.pypots")  # save the model for future use

    ## loading
    # load_saits = SAITS(n_steps=37, n_features=37, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1)
    # load_saits.load("saved_imputer/saits_physionet2012_ep100.pypots")  # reload the serialized model file for following imputation or training

    # imputation = load_saits.impute(dataset)
    # indicating_mask = np.isnan(X.cpu().numpy()) ^ np.isnan(X_ori.cpu().numpy())
    # mae = calc_mae(imputation, np.nan_to_num(X_ori.cpu().numpy()), indicating_mask)
    # print(mae)

sys.exit(0)
######################################################################################