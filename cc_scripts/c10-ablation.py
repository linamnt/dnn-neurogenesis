# imports
import neurodl.cnn as cnn
import numpy as np
import pandas as pd
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
parser.add_argument(
    "-e", "--epochs", help="number of epochs to train", type=int, default=15
)
parser.add_argument(
    "-r", "--repeats", help="number of epochs to train", type=int, default=20
)
parser.add_argument(
    "-x", "--excite", help="whether to excite new neurons", type=bool, default=False
)
parser.add_argument(
    "-t", "--targetted", help="whether to use targetted ablation", type=bool, default=False
)


args = parser.parse_args()

# define paths
SRC_DIR = Path(args.srcdir)
TMP_DIR = Path(args.datadir)
DATA_DIR = TMP_DIR / "data"
RESULTS_DIR = Path(SRC_DIR / "output")
RESULTS_DIR.mkdir(exist_ok=True)
RUNSCRIPTS_DIR = RESULTS_DIR / "run_scripts"
RUNSCRIPTS_DIR.mkdir(exist_ok=True)
EXP_RESULTS = RESULTS_DIR / "results"
EXP_RESULTS.mkdir(exist_ok=True)

RUN_SCRIPT_NAME = f"c10-ablation-{START_TIME}.py"

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / RUN_SCRIPT_NAME)

# IMPORTANT PARAMETERS
REPEATS = args.repeats
EPOCHS = args.epochs
BATCH_SIZE = 4
FREQUENCY = 640
dtype = torch.float
LR = 0.0002
TURNOVER = True
NEUROGENESIS = 8

group_config = {
    "Control": {
        "epochs": EPOCHS,
        "neurogenesis": 0,
        "early_stop": False,
        "lr": LR,
        },
    "Neurogenesis": {
        "epochs": EPOCHS,
        "neurogenesis": NEUROGENESIS,
        "turnover": TURNOVER,
        "frequency": FREQUENCY,
        "targeted_portion": .2,
        "end_neurogenesis": 10,
        "early_stop": False,
        "lr": LR,
        },
}

results = pd.DataFrame(
    index=range(REPEATS * len(group_config)),
    columns=list(range(EPOCHS)) + ["Group", "Test Accuracy", "Repeat"],
)

counter = 0

mode = 'targetted' if args.targetted else 'random'

# DATASET
data_loader = cnn.Cifar10_data(mode="test", batch_size=BATCH_SIZE)
dfs = []
for i in range(REPEATS):
    print(f"Running trial {i+1}/{REPEATS}")
    net = cnn.NgnCnn(args.neurons, seed=i, excite=args.excite)
    for group in group_config:
        net_copy = copy.deepcopy(net)
        net_copy.to(device)
        parameters = group_config[group]
        log = cnn.train_model(
            model=net_copy,
            dataset=data_loader,
            dtype=dtype,
            device=device,
            **parameters,
        )
        
        # accuracy['Accuracy'][0]
        accuracy = cnn.predict(net_copy, data_loader)['Accuracy'][0]
        start = cnn.predict(net_copy, data_loader, train=True)['Accuracy'][0]
        ablate = cnn.ablation(net_copy, data_loader, mode=mode)

        results = pd.DataFrame(ablate, columns=['Proportion', 'Ablation Accuracy'])
        end = results.iloc[-1]['Ablation Accuracy']
        results['Group'] = group
        results['Normalized Ablation Accuracy'] = (results['Ablation Accuracy'] - end)/(start - end)
        results['Test Accuracy'] = accuracy
        dfs.append(results)
        
        final = pd.concat(dfs)
        final.to_csv(
            EXP_RESULTS / "{}.csv".format(RUN_SCRIPT_NAME)
        )


print("Done")
