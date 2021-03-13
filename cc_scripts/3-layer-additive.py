from neurodl import mlp
import pandas as pd
import datetime
import torch
from torch import nn
import numpy as np
from pathlib import Path
import os
import argparse


NUM_WORKERS = os.environ["SLURM_NTASKS"]
data = mlp.load_data(batch_size=50, data_dir="./data")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
args = {
    "Full-None":
    # normal
    {"layer_size": 250, "dataset": data, "dropout": 0},  # DG, CA3
    # dropin
    "Small-Growth": {
        "layer_size": 150,  # DG, CA3
        "dataset": data,
        "dropout": 0,
        "add_new": True,
    },
    "Small-None": {"layer_size": 150, "dataset": data, "dropout": 0},
}


args_new_neuron = {"dropin": 0, "pnew": 10, "replace": 0, "layers": [0]}

lr = 0.001

criterion = nn.NLLLoss()


def main():
    time = datetime.datetime.now().strftime("%Y-%m-%d")
    rand = np.random.randint(100000)
    epochs = 10
    results = pd.DataFrame(
        index=range(len(args)),
        columns=list(range(epochs)) + ["Group", "Training-Length", "Test-Time"],
    )

    counter = 0
    for group, vals in args.items():
        neuralnet = mlp.NgnMlp(**args[group])
        if group == "Small-Growth":
            lr = 0.001
            weights = (True,)
            new = args_new_neuron
            each = False
        else:
            lr = 0.001
            weights = False
            new = None
            each = False
        neuralnet.to(device)
        optimizer = optim.SGD(neuralnet.parameters(), lr)
        criterion = nn.NLLLoss()
        log = mlp.train_model(
            neuralnet,
            optimizer,
            criterion,
            epochs,
            5,
            weights,
            new,
            each,
            early_stop=False,
        )
        # test time accuracy
        loss, acc = mlp.predict(neuralnet, criterion, True)

        results.iloc[counter] = list(log) + [
            group,
            len(np.array(log).nonzero()[0]),
            acc,
        ]
        counter += 1
        print(group, acc)

    results.to_csv("~/{}-{}-{}-2l-additive.csv".format(time, rand, group))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', 
                        help='path to slurm temp dir with data',
                        type=str)
    parser.add_argument('-s', '--srcdir',
                        help='path to source directory files',
                        type=str)
    # define paths
#    RESULTS_PATH = Path("./"

#    main()
