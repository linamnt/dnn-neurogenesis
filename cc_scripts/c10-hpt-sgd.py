import os
from shutil import copyfile
import torch
import numpy as np
import neurodl.cnn as cnn
import argparse
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from ax.service.managed_loop import optimize
from ax import save
import joblib

torch.manual_seed(12345)
datatype = torch.float
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1000

data_loader = cnn.Cifar10_data(mode="validation", batch_size=BATCH_SIZE)

parametrization = [
    {
        "name": "lr",
        "type": "range",
        "bounds": [0.00001, 0.001],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        "log_scale": True,
    },
    {
        "name": "momentum",
        "type": "range",
        "bounds": [0, 1],
        "value_type": "float",  # Optional, defaults to inference from type of "bounds".
    },
    {"name": "epochs", "type": "range", "bounds": [10, 25], "value_type": "int"},
]


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
        model=net,
        dataset=train_loader,
        optim_fn=optim.Adam,
        early_stop=False,
        epochs=parameters["epochs"],
        optim_args={"lr": parameters["lr"], "momentum": parameters["momentum"]},
    )
    return


def evaluate(net, valid_loader, device):
    return cnn.predict(
        model=net, dataset=valid_loader, device=device, valid=True, get_loss=True,
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
        "-t", "--trials", help="number of trials for ax to train", type=int, default=5
    )

    args = parser.parse_args()

    def train_evaluation(parameters, nfolds=5):
        net = cnn.NgnCnn(args.neurons, seed)
        net.to(DEVICE)
        train(net, data_loader, datatype, DEVICE, parameters)
        return evaluate(net, data_loader, DEVICE)

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

    best_parameters, values, experiment, model = optimize(
        parameters=parametrization,
        evaluation_function=train_evaluation,
        objective_name="Accuracy",
        total_trials=args.trials,
    )

    print(
        f"Means: {values[0]}\
        Covariances: {values[1]}"
    )

    print(best_parameters)

    save_name = str(EXP_RESULTS / f"exp-hpt-sgd-{START_TIME}.json")
    save(experiment, save_name)
