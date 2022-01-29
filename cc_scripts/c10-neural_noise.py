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
    "-e", "--epochs", help="number of epochs to train", type=int, default=13
)

args = parser.parse_args()

# define paths
SRC_DIR = Path(args.srcdir)
TMP_DIR = Path(args.datadir)
DATA_DIR = TMP_DIR
RESULTS_DIR = Path(SRC_DIR / "output")
RESULTS_DIR.mkdir(exist_ok=True)
RUNSCRIPTS_DIR = RESULTS_DIR / "run_scripts"
RUNSCRIPTS_DIR.mkdir(exist_ok=True)
EXP_RESULTS = RESULTS_DIR / "results"
EXP_RESULTS.mkdir(exist_ok=True)

run_script_name = f"c10-noise-{START_TIME}.py"

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / run_script_name)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / run_script_name)

# IMPORTANT PARAMETERS
EPOCHS = args.epochs
BATCH_SIZE = 4 
dtype = torch.float
LR = 0.0002
TURNOVER = True
NEUROGENESIS = 8 
TARGETED_PORTION=.2
FREQUENCY = 640 

group_config = {
     "Neurogenesis + Neural Noise": {
        "epochs": EPOCHS,
        "neurogenesis": NEUROGENESIS,
        "turnover": TURNOVER,
       # "targeted_portion": TARGETED_PORTION,
        "frequency": FREQUENCY,
        "end_neurogenesis": 10,
        "early_stop": False,
        "optim_args":{
            "lr": LR,
          #  "momentum": 0.8
        },
    },  
    "Neural Noise": {
        "epochs": EPOCHS, 
        "neurogenesis": 0, 
        "early_stop": False, 
        "optim_args":{
            "lr": LR,
          #  "momentum": 0.8
        }
    },


}

# DATASET
data_loader = cnn.Cifar10_data(mode="validation", data_folder=DATA_DIR, batch_size=BATCH_SIZE, num_workers=16)
REPEATS = 20 

results = pd.DataFrame(index=range(REPEATS*len(group_config)),
                       columns=['Group', 'Test Accuracy', 'Repeat'])
counter = 0
for i in range(REPEATS):
    print(f"Running trial {i+1}/{REPEATS}")
    net = cnn.NgnCnn(args.neurons, channels=3, seed=i,
                    neural_noise=(-0.2,1))
    for group in group_config:
        net_copy = copy.deepcopy(net)
        net_copy.to(device)
        parameters = group_config[group]
        log, optimizer = cnn.train_model(
            model=net_copy,
            dataset=data_loader,
            dtype=dtype,
            device=device,
            **parameters,
        )

        accuracy = cnn.predict(net_copy, data_loader, device=device, valid=True)
        results.iloc[counter] = [group, accuracy['Accuracy'][0], i]
        print(group, accuracy['Accuracy'][0])
        print(log)
        counter += 1

results.to_csv(
    EXP_RESULTS
    / "c10-noise-{}.csv".format(START_TIME)
)

print("Average Accuracy")
print("Done")
