import os
from shutil import copyfile
import torch
import numpy as np
import neurodl.mlp as mlp
import argparse
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
import pandas as pd
from ax import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--datadir", help="path to slurm temp dir with data", type=str
)
parser.add_argument("-s", "--srcdir", help="path to source directory files", type=str)
parser.add_argument(
    "-n", "--neurons", help="number of neurons per layer", type=int, default=250
)
parser.add_argument(
    "-t", "--trials", help="number of trials for ax to train", type=int, default=10
)
args = parser.parse_args()

torch.manual_seed(23456)
datatype = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100
# Number of trials per batch
NUM_BATCH = 5  # TODO

START_TIME = datetime.now().isoformat(timespec="minutes")

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

run_script_name = f"sgd-hpt-{START_TIME}.py"

print("Ax HPT Experiment runscript written to:", RUNSCRIPTS_DIR / run_script_name)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / run_script_name)

data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)


def train(net, optimizer, criterion, parameters):
    """
    Simplified training function for ax optimization.
    """
    epochs = parameters["epochs"]
    opt_args = dict(parameters)

    del opt_args["epochs"]

    mlp.train_model(
        model=net,
        opt_fn=optimizer,
        epochs=epochs,
        criterion=criterion,
        opt_args=opt_args,
        early_stop=False,
    )
    return


def evaluate(model, criterion):
    return mlp.predict(net=model, criterion=criterion, test=True)


def train_evaluation(parameters):
    nfolds = 3  # TODO
    results = np.zeros(nfolds)
    criterion = nn.NLLLoss()

    for i in range(nfolds):
        net = mlp.NgnMlp(data_loader, layer_size=args.neurons, hidden=2)
        net.to(DEVICE)

        optimizer = optim.SGD

        train(
            net, optimizer, criterion, parameters,
        )
        result = evaluate(net, criterion,)
        results[i] = result[0]

    return {"Cross Entropy Loss": (results.mean(), results.std())}


search_space = SearchSpace(
    parameters=[
        RangeParameter(  
            name="epochs", lower=10, upper=20, parameter_type=ParameterType.INT
        ),
        RangeParameter(
            name="momentum", lower=0, upper=1, parameter_type=ParameterType.FLOAT
        ),
        RangeParameter(
            name="lr",
            lower=0.000001,
            upper=1,
            log_scale=True,
            parameter_type=ParameterType.FLOAT,
        ),
    ]
)


experiment = SimpleExperiment(
    name="neurogenesis_hpt",
    search_space=search_space,
    evaluation_function=train_evaluation,
    objective_name="Cross Entropy Loss",
    minimize=True,
)

sobol = Models.SOBOL(search_space=experiment.search_space, experiment=experiment,)


print("Running initial SOBOL trials.")

experiment.new_batch_trial(generator_run=sobol.gen(NUM_BATCH))
save_name = str(EXP_RESULTS / f"exp-bo-mlp-sgd-{START_TIME}.json")

for i in range(args.trials):
    print(
        f"[{datetime.now().isoformat(timespec='minutes')[-10:]}] Trial {i+1}/{args.trials}"
    )
    intermediate_gp = Models.GPEI(experiment=experiment, data=experiment.eval())
    save(experiment, save_name)
    experiment.new_batch_trial(generator_run=intermediate_gp.gen(NUM_BATCH))

experiment.eval()
print("Done!")
save(experiment, save_name)
