"""
Do networks with optimal neurogenesis cluster better?
"""
# imports
import neurodl.mlp as mlp
import neurodl.tsne as tsne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torch
import seaborn as sns
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
    "-r", "--repeats", help="number of epochs to train", type=int, default=3
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

RUN_SCRIPT_NAME = f"mlp-tsne_plots{START_TIME}.py"

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)

# IMPORTANT PARAMETERS
REPEATS = args.repeats
EPOCHS = 30
LR = 0.05
BATCH_SIZE = 100
dtype = torch.float

group_config = {
    "Control": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": False,
            "early_stop": False,
        },
        "optimizer_args": {"lr": 0.03, "momentum": 0.8},
    },
    "Low": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": True,
            "early_stop": False,
            "new_args": {"pnew": 0.1, "replace": True, "layers": [0],},
        },
        "optimizer_args": {"lr": LR, "momentum": 0.8},
    },
    "Optimal": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": True,
            "early_stop": False,
            "new_args": {"pnew": 0.3, "replace": True, "layers": [0],},
        },
        "optimizer_args": {"lr": LR, "momentum": 0.8},
    },
    "High": {
        "training_args": {
            "opt_fn": optim.SGD,
            "epochs": EPOCHS,  # TODO
            "neurogenesis": True,
            "early_stop": False,
            "new_args": {"pnew": 0.85, "replace": True, "layers": [0],},
        },
        "optimizer_args": {"lr": LR, "momentum": 0.8},
    },
}

results = pd.DataFrame(
    index=range(REPEATS * len(group_config)),
    columns=["Group", "KLD", "Silhouette", "Accuracy"],
)

counter = 0

# DATASET
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)


## CODE OF INTEREST
for i in range(args.repeats):
    print(f"Trial {i+1}/{args.repeats}")
    for group in group_config:
        net = mlp.NgnMlp(data_loader)
        net.to(device)

        criterion = nn.NLLLoss()

        parameters = group_config[group]["training_args"]

        log = mlp.train_model(
            model=net,
            full_set=True,
            opt_args=group_config[group]["optimizer_args"],
            criterion=criterion,
            **parameters,
        )
        loss, acc = mlp.predict(net, criterion, test=True)
        middle, targets = net.return_layer(0, "Train")
        if not i:
            to_file = group
        else:
            to_file = None
        # run tsne pipe on train
        tsne.tsne_pipe(
            middle,
            targets,
            plot=True,
            shape=1024,
            to_file=to_file if to_file is None else EXP_RESULTS / (to_file + ".png"),
        )
        plt.close("all")
        # on test
        middle, targets = net.return_layer(0, "Train")

        kld, silhouette = tsne.tsne_pipe(
            middle,
            targets,
            plot=True,
            shape=1024,
            to_file=to_file
            if to_file is None
            else EXP_RESULTS / (to_file + "test.png"),
        )
        plt.close("all")
        # save to df
        print(results)
        results.iloc[counter] = [group, kld, silhouette, acc]
        counter += 1


results.to_csv(EXP_RESULTS / "{}-{}.csv".format(RUN_SCRIPT_NAME, START_TIME))  # TODO


## END

print("Done")
