"""
Do networks with neurogenesis do better than networks with dropout?
"""
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

torch.manual_seed(23456)
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
# )
parser.add_argument(
    "-r", "--repeats", help="number of epochs to train", type=int, default=20
)
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

RUN_SCRIPT_NAME = f"mlp-dropout-{START_TIME}.py"

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)

# IMPORTANT PARAMETERS
REPEATS = args.repeats
EPOCHS = 45
DROPOUT_EPOCHS = 65
BATCH_SIZE = 100
LR = 0.001
dtype = torch.float

group_config = {
    "Neurogenesis": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": True,
            "end_neurogenesis": 10,
            "early_stop": False,
            "freq": 200,
          #  "targeted_portion": 0.20,
            "new_args": {"pnew": 0.025, "replace": True, "layers": [1],},
        },
        "optimizer_args": {"lr": LR,}# 'momentum': 0.8},
    },
    # "Control": {
    #     "training_args": {
    #         "opt_fn": optim.SGD,
    #         "epochs": EPOCHS,  # TODO
    #         "neurogenesis": 0,
    #         "early_stop": False,
    #     },
    #     "optimizer_args": {"lr": LR,}# 'momentum': 0.8},
    # },
    "Excite": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": True,
            "end_neurogenesis": 15,
            "early_stop": False,
            "freq": 500,
          #  "targeted_portion": 0.20,
            "new_args": {"pnew": 0.05, "replace": True, "layers": [0],},
        },
        "optimizer_args": {"lr": LR,}# 'momentum': 0.8},
    },   
    # "L2": {
    #     "training_args": {
    #         "opt_fn": optim.SGD,
    #         "epochs": EPOCHS,  # TODO
    #         "neurogenesis": 0,
    #         "early_stop": False,
    #     },
    #     "optimizer_args": {"lr": LR, "weight_decay": 0.01,}# 'momentum': 0.8'},
    # },
    # "Dropout": {
    #     "training_args": {
    #         "opt_fn": optim.SGD,
    #         "epochs": DROPOUT_EPOCHS,  # TODO
    #         "neurogenesis": 0,
    #         "early_stop": False,
    #     },
    #     "optimizer_args": {"lr": 0.05,}# 'momentum': 0.8},
    # },
}

results = pd.DataFrame(
    index=range(REPEATS * len(group_config)),
    columns=list(range(DROPOUT_EPOCHS)) + ["Group", "Test Accuracy", "Repeat"],
)

counter = 0

# DATASET
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)

for repeat in range(REPEATS):
    print(f"Running trial {repeat+1}/{REPEATS}")
    net = mlp.NgnMlp(data_loader, control=True)
    for group in group_config:
        net_copy = copy.deepcopy(net)
        net_copy.to(device)
        if group == "Excite":
            net_copy.excite = True
            net_copy.modified_layers = [1]
        if group == "Neurogenesis":
            net_copy.excite = True
            net_copy.control = False
        criterion = nn.NLLLoss()

        parameters = group_config[group]["training_args"]

        log = mlp.train_model(
            model=net_copy,
            full_set=True,
            opt_args=group_config[group]["optimizer_args"],
            criterion=criterion,
            eval=5,
            **parameters,
        )

        avg_loss, accuracy = mlp.predict(net_copy, criterion=criterion, test=True)

        log_diff = DROPOUT_EPOCHS - len(log)
        if log_diff:
            log = log + [0] * log_diff
        results.iloc[counter] = log + [group, accuracy, repeat]

        counter += 1

        results.to_csv(
            EXP_RESULTS / "{}-{}.csv".format(RUN_SCRIPT_NAME, START_TIME)
        )  # TODO

print("Average Accuracy")
print("Done")
