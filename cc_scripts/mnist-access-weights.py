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
DATA_DIR = Path(args.datadir)
RESULTS_DIR = Path(SRC_DIR / "output")
RESULTS_DIR.mkdir(exist_ok=True)
RUNSCRIPTS_DIR = RESULTS_DIR / "run_scripts"
RUNSCRIPTS_DIR.mkdir(exist_ok=True)
EXP_RESULTS = RESULTS_DIR / "results"
EXP_RESULTS.mkdir(exist_ok=True)
JOB_NAME = "mlp-compare-step-size" # TODO

run_script_name = f"{JOB_NAME}-{START_TIME}.py"

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / run_script_name)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / run_script_name)

# IMPORTANT PARAMETERS
REPEATS = args.repeats
EPOCHS = 5
BATCH_SIZE = 500
dtype = torch.float
TURNOVER = 1.0
NEUROGENESIS = 50

configs = {
    "Model": {
        "optimizer": optim.Adam_mod,
        "training_args": {
            "epochs": EPOCHS,
             "neurogenesis": 0,
             "early_stop": False,
            },
        "optimizer_args": {
            "lr": 0.00055,
            },
    },
    "Neurogenesis": {
        "neurogenesis": NEUROGENESIS,
        "layers": [0],
        "replace": TURNOVER,
        },
    },
}

results = pd.DataFrame(
    index=range(REPEATS)),
    columns=list(range(EPOCHS)) + ["Group", "Test Accuracy", "Repeat"],
)

counter = 0

# DATASET
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)

for repeat in range(REPEATS):
    print(f"Running trial {repeat+1}/{REPEATS}")
    net_copy = mlp.NgnMlp(data_loader, layer_size=args.neurons, hidden=3)
    net_copy.to(device)
    criterion = nn.NLLLoss()

    parameters = configs['Model']['training_args']

    optimizer = configs['Model']['optimizer'](
        net.parameters(),
        **configs['Model']['optimizer_args'])


    log, _ = mlp.train_model(
        model=net_copy,
        opt=optimizer,
        criterion=criterion,
        **parameters,
    )

    net_copy.add_new(**configs['Neurogenesis'])

    log, optimizer = mlp.train_model(
        model=net_copy,
        opt=optimizer,
        criterion=criterion,
        **parameters,

    optimizer
    )

    accuracy = mlp.predict(net_copy, criterion=criterion, test=True)

    log_diff = EPOCHS - len(log)
    if log_diff:
        log = log + [0]*log_diff
    results.iloc[counter] = log + [group, accuracy[0], repeat]

    counter += 1


results.to_csv(
    EXP_RESULTS
    / "{}-{}.csv".format( # TODO
        JOB_NAME,
        START_TIME
    )
)

print("Average Accuracy")
results['Test Accuracy'] = results['Test Accuracy'].astype(float)
print(results.groupby("Group")["Test Accuracy"].mean())
print("Done")
