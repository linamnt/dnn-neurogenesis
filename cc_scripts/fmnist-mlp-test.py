# imports
import neurodl.mlp as mlp
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torch
import argparse
import copy

# file management
import os
from shutil import copyfile
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

START_TIME = datetime.now().isoformat(timespec="minutes")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--datadir", help="path to slurm temp dir with data", type=str
)
parser.add_argument("-s", "--srcdir", help="path to source directory files", type=str)
parser.add_argument(
    "-n", "--neurons", help="number of neurons per layer", type=int, default=250
)
# parser.add_argument(
#     "-e", "--epochs", help="number of epochs to train", type=int, default=11
#

args = parser.parse_args()

# define paths
SRC_DIR = Path(args.srcdir)
DATA_DIR = Path(args.datadir) 
RESULTS_DIR = Path(SRC_DIR / "output")
RESULTS_DIR.mkdir(exist_ok=True)
RUNSCRIPTS_DIR = RESULTS_DIR / "run_scripts"
RUNSCRIPTS_DIR.mkdir(exist_ok=True)
EXP_RESULTS = RESULTS_DIR / "results"
EXP_RESULTS.mkdir(exist_ok=True)


# IMPORTANT PARAMETERS
EPOCHS = 15
LR = 0.0001
BATCH_SIZE = 100 
dtype = torch.float

group_config = {
    "Neurogenesis": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": True,
            "end_neurogenesis": 20,
            "early_stop": False,
            "freq": 100,
            "targeted_portion": 0.10,
            "new_args": {"pnew": 0.02,
               # "excite":.4,
                "replace": True, "layers": [1]},
        },
        "optimizer_args": {"lr": 0.001,} #"momentum": 0.95},
    },
    "Control": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": 0,
            "early_stop": False,
        },
        "optimizer_args": {"lr": 0.0001,}# "momentum": 0.95},
    },
    "Dropout": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": 60,  # TODO
            "neurogenesis": 0,
            "early_stop": False,
        },
        "optimizer_args": {"lr": 0.001, }# "momentum": 0.8},
    },
}


# DATASET
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, c10=True, num_workers=16, split=0.85
)

net = mlp.NgnMlp(data_loader, c10=True)
for group in group_config:
    net_copy = copy.deepcopy(net)
    net_copy.to(device)

    if group == 'Dropout':
        net_copy.dropout = 0.15
        
    criterion = nn.NLLLoss()

    parameters = group_config[group]["training_args"]
    log = mlp.train_model(
        model=net_copy,
        full_set=False,
        opt_args=group_config[group]["optimizer_args"],
        criterion=criterion,
        eval=5,
        **parameters,
    )

    avg_loss, accuracy = mlp.predict(net_copy, criterion=criterion, test=False)

    log_diff = EPOCHS - len(log)
    print(group)
    print(log)
    print("Test Accuracy:", accuracy)
