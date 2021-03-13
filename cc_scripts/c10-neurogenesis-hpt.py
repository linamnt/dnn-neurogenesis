import os
from shutil import copyfile
import torch
import numpy as np
import neurodl.cnn as cnn
import argparse
import torch.optim as optim
from datetime import datetime
from pathlib import Path
import pandas as pd
from ax import *

torch.manual_seed(23456)
datatype = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
LEARNING_RATE = 0.0002
data_loader = cnn.Cifar10_data(mode="validation", batch_size=BATCH_SIZE)


def train(net, train_loader, dtype, device, parameters):
    """
    Simplified training function for ax optimization.

    Args:
        net: Neural network object.
        train_loader: Training data loader.
        parameters (dict): Dictionary of parameter values.
        dtype: Data type.
        device: Device specifying cpu or gpu training.
    """
    cnn.train_model(
        model=net, dataset=train_loader, dtype=dtype, device=device,
        optim_args={"lr": parameters["lr"]},
        **parameters
    )
    return


def evaluate(net, valid_loader, device):
    return cnn.predict(
        model=net, dataset=valid_loader, device=device, valid=True, get_loss=True,
    )


def train_evaluation(parameters):  # TODO
    nfolds = 3
    results = np.zeros(nfolds)

    for i in range(nfolds):
        net = cnn.NgnCnn(args.neurons, seed=i)
        net.to(DEVICE)
        train(net, data_loader, datatype, DEVICE, parameters)
        result = evaluate(net, data_loader, DEVICE)
        results[i] = result["Loss"][0]

    return {"cross_entropy": (results.mean(), results.std())}


search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="neurogenesis", lower=0, upper=50, parameter_type=ParameterType.INT
        ),
        RangeParameter(
            name="frequency", lower=1, upper=5000, parameter_type=ParameterType.INT
        ),
        RangeParameter(
            name="turnover", lower=0.0, upper=1.0, parameter_type=ParameterType.FLOAT
        ),
        FixedParameter(
            name="end_neurogenesis", value=10, parameter_type=ParameterType.INT
        ),
        FixedParameter(
            name="early_stop", value=False, parameter_type=ParameterType.BOOL
        ),
        FixedParameter(
            name="lr", value=LEARNING_RATE, parameter_type=ParameterType.FLOAT
        ),
        FixedParameter(
            name="epochs", value=15, parameter_type=ParameterType.INT
        ),  # TODO
    ]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--datadir", help="path to slurm temp dir with data", type=str
    )
    parser.add_argument(
        "-s", "--srcdir", help="path to source directory files", type=str
    )
    parser.add_argument(
        "-n", "--neurons", help="number of neurons per layer", type=int, default=250
    )
    parser.add_argument(
        "-t", "--trials", help="number of trials for ax to train", type=int, default=10
    )

    args = parser.parse_args()

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

    run_script_name = f"runscript-{START_TIME}.py"

    print("Ax HPT Experiment runscript written to:", RUNSCRIPTS_DIR / run_script_name)
    copyfile(os.path.realpath(__file__), RUNSCRIPTS_DIR / run_script_name)

    experiment = SimpleExperiment(
        name="neurogenesis_hpt",
        search_space=search_space,
        evaluation_function=train_evaluation,
        objective_name="cross_entropy",
        minimize=True,
    )

    experiment.status_quo = Arm(
        parameters={
            "neurogenesis": 0,
            "frequency": 0,
            "turnover": 0,
            "end_neurogenesis": 11,
            "early_stop": False,
            "lr": LEARNING_RATE,
            "epochs": 11,
        }
    )

    sobol = Models.SOBOL(search_space=experiment.search_space, experiment=experiment,)
    NUM_BATCH = 10  # TODO

    print("Running initial SOBOL trials.")

    experiment.new_batch_trial(generator_run=sobol.gen(NUM_BATCH))
    save_name = str(EXP_RESULTS / f"exp-bo-ngn-{START_TIME}.json")

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
