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
    "-t", "--trials", help="number of trials for ax to train", type=int, default=5,
)
args = parser.parse_args()

# SET RANDOM SEED
torch.manual_seed(23456)

# PARAMETERS/CONFIG VALUES
datatype = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1000
EPOCHS = 35
LR = 0.00055  # TODO
# Number of trials per batch
NUM_BATCH = 10  # TODO

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

# SAVE EXPERIMENT RUNSCRIPT
run_script_name = f"runscript-mlp-neurogenesis-hpt{START_TIME}.py"

print("Ax HPT Experiment runscript written to:", RUNSCRIPTS_DIR / run_script_name)
copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / run_script_name)

# LOAD DATA
data_loader = mlp.load_data(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, num_workers=16, split=0.85
)


def train(net, optimizer, opt_args, criterion, parameters):
    """
    Simplified training function for ax optimization.
    """
    new_args = {
        "pnew": parameters["pnew"],
        "replace": True,
        "layers": [parameters["layers"]] if parameters["layers"] < 2 else [0, 1],
    }
    mlp.train_model(
        model=net,
        opt_fn=optimizer,
        opt_args=opt_args,
        criterion=criterion,
        neurogenesis=True,
        new_args=new_args,
    )
    return


def evaluate(net, criterion):
    return mlp.predict(net=net, criterion=criterion, test=True)


def train_evaluation(parameters):
    nfolds = 3  # TODO
    results = np.zeros(nfolds)
    criterion = nn.NLLLoss()

    print(parameters)
    for i in range(nfolds):
        net = mlp.NgnMlp(data_loader)
        net.to(DEVICE)

        optimizer = optim.SGD
        opt_args = {"lr": parameters["lr"];
                    "momentum": parameters["momentum"]}

        train(
            net, optimizer, opt_args, criterion, parameters,
        )
        result = evaluate(net, criterion,)
        results[i] = result[0]
    print(parameters)
    return {"Cross Entropy Loss": (results.mean(), results.std())}


search_space = SearchSpace(
    parameters=[
        RangeParameter(  # TODO
            name="pnew", lower=0, upper=100, parameter_type=ParameterType.INT
        ),
        ChoiceParameter(
            name="layers", parameter_type=ParameterType.INT, values=[0, 1, 2]
        ),
        FixedParameter(
            name="lr", value=LR, parameter_type=ParameterType.FLOAT,  # TODO
        ),
        FixedParameter(name="epochs", value=EPOCHS, parameter_type=ParameterType.INT),
        FixedParameter(
            name="early_stop", value=False, parameter_type=ParameterType.BOOL
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
save_name = str(EXP_RESULTS / f"exp-bo-mlp-ngn-{START_TIME}.json")

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
