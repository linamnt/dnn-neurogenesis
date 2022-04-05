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
    "-e", "--epochs", help="number of epochs to train", type=int, default=14
)
parser.add_argument(
    "-r", "--repeats", help="number of epochs to train", type=int, default=20
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

run_script_name = f"c10-weights-{START_TIME}.py"

# save a copy of the runscript
print("Runscript written to:", RUNSCRIPTS_DIR / run_script_name)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / run_script_name)

# IMPORTANT PARAMETERS
REPEATS = 1
EPOCHS = args.epochs
BATCH_SIZE = 4
FREQUENCY = 640
dtype = torch.float
LR = 0.0002
TURNOVER = True
NEUROGENESIS = 8

group_config = {
        "Control": {"epochs": EPOCHS, "neurogenesis": 0, "early_stop": False, "lr": LR,},
    "Neurogenesis": {
        "epochs": EPOCHS,
        "neurogenesis": NEUROGENESIS,
        "turnover": TURNOVER,
        "frequency": FREQUENCY,
        "end_neurogenesis": 10,
        "early_stop": False,
        "lr": LR,
    },
    "Dropout": {"epochs": EPOCHS+7, "neurogenesis": 0, "early_stop": False, "lr": LR,},

}



results = pd.DataFrame(
    index=range(REPEATS * len(group_config) * 200 ),
    columns= list(range(250)) + ['Label', 'Group']
)

counter = 0

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

# DATASET
data_loader = cnn.Cifar10_data(mode="test", batch_size=4)

for i in range(REPEATS):
    print(f"Running trial {i+1}/{REPEATS}")
    net = cnn.NgnCnn(args.neurons, seed=i)
    for group_num, group in enumerate(group_config):
        net_copy = copy.deepcopy(net)
        if "Dropout" in group:
            net_copy.dropout = 0.2
        net_copy.to(device)
        parameters = group_config[group]
        log, _ = cnn.train_model(
            model=net_copy,
            dataset=data_loader,
            dtype=dtype,
            device=device,
            **parameters,
        )

        loader = data_loader.test
        data = [(img, lbl) for img, lbl in loader]
        reps = np.zeros([200,250])
        labels = np.zeros(200)
        for ix in range(50):
            images, labels = data[i]
            images = images.to(device)
            outputs = net_copy(images, extract_layer=1)
            outputs = outputs.cpu().detach().numpy()
            print(outputs.shape)
            results.iloc[group_num*200+ix*4:group_num*200+ix*4+4,0:250] = outputs
            results.iloc[group_num*200+ix*4:group_num*200+ix*4+4,250] = labels
            results.iloc[group_num*200+ix*4:group_num*200+ix*4+4,251] = [group]*4

        results.to_csv(
                EXP_RESULTS
                / "c10-selectivity-{}.csv".format(START_TIME)
            )



results.to_csv(
    EXP_RESULTS
    / "c10-selectivity-{}.csv".format(START_TIME)
)
