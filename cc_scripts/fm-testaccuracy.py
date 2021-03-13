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
    "-e", "--epochs", help="number of epochs to train", type=int, default=15
)

args = parser.parse_args()

# define paths
SRC_DIR = Path(args.srcdir)
DATA_DIR = Path(args.datadir)

# IMPORTANT PARAMETERS
EPOCHS = args.epochs
BATCH_SIZE =4
dtype = torch.float
LR = 0.0002
TURNOVER = True
NEUROGENESIS =8
TARGETED_PORTION=.3
FREQUENCY = 200 

group_config = {
     "Neurogenesis": {
        "epochs": EPOCHS,
        "neurogenesis": NEUROGENESIS,
        "turnover": TURNOVER,
       # "targeted_portion": TARGETED_PORTION,
        "frequency": FREQUENCY,
        "end_neurogenesis":18,
        "early_stop": False,
        "optim_args":{
        "lr": LR,},
    },  
     "Control": {
        "epochs": EPOCHS, 
        "neurogenesis": 0, 
        "early_stop": False, 
        "optim_args":{
            "lr": LR,
        }
    },
     "Dropout": {
        "epochs": 20, 
        "neurogenesis": 0, 
        "early_stop": False, 
        "optim_args":{
            "lr": 0.0002,
        }
    },


}

# DATASET
data_loader = cnn.Cifar10_data(mode="validation",data_folder=DATA_DIR, batch_size=BATCH_SIZE, fashion=True)
REPEATS = 1
for i in range(REPEATS):
    print(f"Running trial {i+1}/{REPEATS}")
    net = cnn.NgnCnn(args.neurons,channels=1,excite=True, seed=i)
    for group in group_config:
        net_copy = copy.deepcopy(net)
        net_copy.to(device)
        if group == 'Dropout':
            net_copy.dropout = 0.25
        parameters = group_config[group]
        log= cnn.train_model(
            model=net_copy,
            dataset=data_loader,
            dtype=dtype,
            device=device,
            **parameters,
        )

        accuracy = cnn.predict(net_copy, data_loader, device=device, valid=False)
        print(group, accuracy['Accuracy'][0])


print("Average Accuracy")
print("Done")
print(log[0])
