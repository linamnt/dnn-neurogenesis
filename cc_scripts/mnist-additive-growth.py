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
filename = "mlp-growth"
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
parser.add_argument(
     "-e", "--epochs", help="number of epochs to train", type=int, default=20
 )
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

run_script_name = f"{filename}-{START_TIME}.py"

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / run_script_name)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / run_script_name)

# IMPORTANT PARAMETERS
REPEATS = args.repeats
EPOCHS = 35 
LR = 0.0006
BATCH_SIZE = 100
dtype = torch.float

group_config = {
    "Neurogenesis": {
        "optimizer": optim.Adam,
        "training_args": {
            "epochs": EPOCHS,  # TODO
            "neurogenesis": True,
            "turnover": False,
            "early_stop": False,
            "freq": 1000,
            "end_neurogenesis": 10,
            "new_args": {"pnew": 10, "replace": False, "layers": [1],},
        },
        "optimizer_args": {"lr": LR,}# "momentum": 0.9},  # TODO
    },
    "Control": {
        "optimizer": optim.Adam,
        "training_args": {
            "epochs": EPOCHS,  # TODO
            "neurogenesis": 0,
            "early_stop": False,
        },
        "optimizer_args": {"lr": LR,}# "momentum": 0.9},
    },
    "Control Small": {
        "optimizer": optim.Adam,
        "training_args": {
            "epochs": EPOCHS,  # TODO
            "neurogenesis": 0,
            "early_stop": False,
        },
        "optimizer_args": {"lr": LR,}# "momentum": 0.9},
    },
}

results = pd.DataFrame(
    index=range(REPEATS * len(group_config)),
    columns=list(range(EPOCHS)) + ["Group", "Test Accuracy", "Repeat"],
)

counter = 0

# DATASET
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)

for repeat in range(REPEATS):
    print(f"Running trial {repeat+1}/{REPEATS}")
    for group in group_config:
        if group == "control":
            net_copy = mlp.NgnMlp(data_loader, layer_size=[250, 250,500])
        elif group == "Control Small":
            net_copy = mlp.NgnMlp(data_loader, layer_size=[250, 150,500])
        else:
            net_copy = mlp.NgnMlp(data_loader, layer_size=[250, 150,500])

        net_copy.to(device)
        criterion = nn.NLLLoss()

        parameters = group_config[group]["training_args"]

        optimizer = group_config[group]["optimizer"]

        log = mlp.train_model(
            model=net_copy,
            full_set=True,
            opt_fn=optimizer,
            opt_args=group_config[group]["optimizer_args"],
            criterion=criterion,
            **parameters,
        )

        avg_loss, accuracy = mlp.predict(net_copy, criterion=criterion, test=True)

        log_diff = EPOCHS - len(log)
        if log_diff:
            log = log + [0] * log_diff
        results.iloc[counter] = log + [group, accuracy, repeat]

        counter += 1


results.to_csv(EXP_RESULTS / f"{filename}-{START_TIME}.csv")  # TODO

print("Average Accuracy")
results["Test Accuracy"] = results["Test Accuracy"].astype(float)
print(results.groupby("Group")["Test Accuracy"].mean())
print("Done")
