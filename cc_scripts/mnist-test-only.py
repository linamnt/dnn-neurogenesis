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
DATA_DIR = Path(args.datadir) / "data"
RESULTS_DIR = Path(SRC_DIR / "output")
RESULTS_DIR.mkdir(exist_ok=True)
RUNSCRIPTS_DIR = RESULTS_DIR / "run_scripts"
RUNSCRIPTS_DIR.mkdir(exist_ok=True)
EXP_RESULTS = RESULTS_DIR / "results"
EXP_RESULTS.mkdir(exist_ok=True)

RUN_SCRIPT_NAME = f"mlp-compare-size-{START_TIME}.py"  # TODO

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)

# IMPORTANT PARAMETERS
EPOCHS = 300
LR = 0.0001
BATCH_SIZE = 100
dtype = torch.float

group_config = {
    "Control": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": 0,
            "early_stop": False,
        },
        "optimizer_args": {"lr": LR, "momentum": 0.9, "weight_decay":0.001},
    }
}


# DATASET
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)

net = mlp.NgnMlp(data_loader)
net.to(device)
for group in group_config:
    criterion = nn.NLLLoss()

    parameters = group_config[group]["training_args"]

    log = mlp.train_model(
        model=net,
        full_set=True,
        opt_args=group_config[group]["optimizer_args"],
        criterion=criterion,
        **parameters,
    )

    avg_loss, accuracy = mlp.predict(net, criterion=criterion, test=True)

    log_diff = EPOCHS - len(log)
    print(group)
    print(log)
    print("Test Accuracy:", accuracy)
