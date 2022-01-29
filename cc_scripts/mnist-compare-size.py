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
    "-r", "--repeats", help="number of epochs to train", type=int, default=5
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

RUN_SCRIPT_NAME = f"mlp-compare-size-{START_TIME}.py"  # TODO

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)

# IMPORTANT PARAMETERS
REPEATS = args.repeats
EPOCHS = 20
BATCH_SIZE = 4
dtype = torch.float

sizes = {"Large": 1000, "Regular": 250}

group_config = {
    "Neurogenesis": {
        "Regular": {
            "training_args": {
                "opt_fn": optim.Adam,
                "epochs": EPOCHS,  # TODO
                "neurogenesis": True,
                "early_stop": False,
                "new_args": {"pnew": 0.40, "replace": True, "layers": [0],},
            },
            "optimizer_args": {"lr": 0.0001,},
        },
        "Large": {
            "training_args": {
                "opt_fn": optim.Adam,
                "epochs": 12,  # TODO
                "neurogenesis": True,
                "end_neurogenesis": 12,
                "early_stop": False,
                "new_args": {"pnew": 7, "replace": True, "layers": [0],},
            },
            "optimizer_args": {"lr": 0.0001,},
        },
    },
    "Control": {
        "Regular": {
            "training_args": {
                "opt_fn": optim.Adam,
                "epochs": EPOCHS,  # TODO
                "neurogenesis": 0,
                "early_stop": False,
            },
            "optimizer_args": {"lr": 0.0001,},
        },
        "Large": {
            "training_args": {
                "opt_fn": optim.Adam,
                "epochs": 12,  # TODO
                "neurogenesis": 0,
                "early_stop": False,
            },
            "optimizer_args": {"lr": 0.0001,},
        },
    },
}

results = pd.DataFrame(
    index=range(REPEATS * 2 * len(group_config)),
    columns=list(range(EPOCHS)) + ["Group", "Test Accuracy", "Repeat", "Size"],
)

counter = 0

# DATASET
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)

for repeat in range(REPEATS):
    print(f"Running trial {repeat+1}/{REPEATS}")
    for size in sizes:
        net = mlp.NgnMlp(data_loader, layer_size=sizes[size], hidden=2)
        for group in group_config:
            net_copy = copy.deepcopy(net)
            net_copy.to(device)
            criterion = nn.NLLLoss()

            parameters = group_config[group][size]["training_args"]

            log = mlp.train_model(
                model=net_copy,
                full_set=True,
                opt_args=group_config[group][size]["optimizer_args"],
                criterion=criterion,
                **parameters,
            )

            avg_loss, accuracy = mlp.predict(net_copy, criterion=criterion, test=True)

            log_diff = EPOCHS - len(log)
            if log_diff:
                log = log + [0] * log_diff
            results.iloc[counter] = log + [group, accuracy, repeat, size]

            counter += 1


results.to_csv(EXP_RESULTS / "{}-{}.csv".format(RUN_SCRIPT_NAME, START_TIME))  # TODO

print("Average Accuracy")
results["Test Accuracy"] = results["Test Accuracy"].astype(float)
print(results.groupby("Group")["Test Accuracy"].mean())
print("Done")
