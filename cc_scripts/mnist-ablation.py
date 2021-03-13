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
    "-r", "--repeats", help="number of epochs to train", type=int, default=10
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

RUN_SCRIPT_NAME = f"mlp-ablation-{START_TIME}.py"  # TODO

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)

# IMPORTANT PARAMETERS
REPEATS = args.repeats
EPOCHS = 21
LR = 0.0001
BATCH_SIZE = 4
dtype = torch.float

group_config = {
    "Neurogenesis": {
        "training_args": {
            "opt_fn": optim.Adam,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": True,
            "end_neurogenesis": 11,
            "early_stop": False,
            "new_args": {"pnew": 0.3, "replace": True, "layers": [0],},
        },
        "optimizer_args": {"lr": 0.03, "momentum": 0.9},
    },
    "Control": {
        "training_args": {
            "opt_fn": optim.Adam,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": 0,
            "early_stop": False,
        },
        "optimizer_args": {"lr": LR, "momentum": 0.9},
    },
    "Dropout": {
        "training_args": {
            "opt_fn": optim.Adam,
            "epochs": 35,  # TODO
            "neurogenesis": 0,
            "early_stop": False,
        },
        "optimizer_args": {"lr": LR, "momentum": 0.9, },
    },
}

dfs = []


counter = 0

# DATASET
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)


for repeat in range(REPEATS):
    print(f"Running trial {repeat+1}/{REPEATS}")
    net = mlp.NgnMlp(data_loader)
    for group in group_config:
        net_copy = copy.deepcopy(net)
        net_copy.to(device)

        if group == "Dropout":
            net_copy.dropout = 0.2

        criterion = nn.NLLLoss()

        parameters = group_config[group]["training_args"]

        log = mlp.train_model(
            model=net_copy,
            full_set=True,
            opt_args=group_config[group]["optimizer_args"],
            criterion=criterion,
            **parameters,
        )

        avg_loss, accuracy = mlp.predict(net_copy, criterion=criterion, test=True)
        _, start = mlp.predict(net_copy, criterion=criterion, test=False)
        ablate = mlp.ablation(net_copy, criterion=criterion, layer=[0])

        results = pd.DataFrame(ablate, columns=["Proportion", "Ablation Accuracy"])
        end = results.iloc[-1]["Ablation Accuracy"]
        results["Group"] = group
        results["Normalized Ablation Accuracy"] = (
            results["Ablation Accuracy"] - end
        ) / (start - end)
        results["Test Accuracy"] = accuracy
        dfs.append(results)
        final = pd.concat(dfs)
        final.to_csv(
            EXP_RESULTS / "{}.csv".format(RUN_SCRIPT_NAME)
        )  # TODO

print("Done")
