import numpy as np
from scipy import stats
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import os
from neurodl.targeted_neurogenesis import targeted_neurogenesis

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(
    mode,
    data_folder="./data",
    num_workers=16,
    batch_size=50,
    split=0.1,
    seed=23,
    fashion=False,
):
    """
    Helper function to read in image dataset, and split into
    training, validation and test sets.
    ===
    mode: str, ['validation', 'test]. If 'validation', training data
         will be divided based on split parameter.
         If test, .valid = None, and all training data is used for training
    split: float, where 0 < split < 1. Where train = split * num_samples
        and valid = (1 - split) * num_samples
    seed: int, random seed to generate validation/training split
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    assert mode in ["validation", "test"]

    if fashion:
        trainset = torchvision.datasets.MNIST(
            data_folder,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        testset = torchvision.datasets.MNIST(
            data_folder,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        print("Loaded FMNIST dataset")
    else:
        trainset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=False, transform=transform
        )

        testset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=False, transform=transform
        )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    if mode == "validation":
        from sklearn.model_selection import train_test_split

        num_train = 50000
        indices = list(range(num_train))

        train_idx, valid_idx = train_test_split(
            indices, test_size=split, random_state=seed
        )

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=train_sampler,
        )

        validloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=valid_sampler,
            drop_last=True,
        )
        print("Created data loaders")
        return trainloader, validloader, testloader

    elif mode == "test":
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )
        print("Created data loaders")
        return trainloader, testloader


class Cifar10_data(object):
    def __init__(
        self,
        mode="validation",
        data_folder="./data",
        batch_size=50,
        fashion=False,
        num_workers=16,
        split=0.1,
        seed=23,
    ):
        if mode == "validation":
            self.train, self.valid, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                batch_size=batch_size,
                num_workers=num_workers,
                split=split,
                fashion=fashion,
                seed=seed,
            )
        elif mode == "test":
            self.train, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                seed=seed,
                fashion=fashion,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            self.valid = None


def early_stopping(starting, patience, count, best_score, prediction):
    # starting accuracy (in case network is not training at all)
    if starting is None:
        starting = prediction["Accuracy"][0]
    # first epoch
    if best_score is None:
        best_score = prediction["Loss"][0]
    # if score is decreasing, start counter
    elif np.round(prediction["Loss"][0], 4) < best_score:
        count = 0
        best_score = prediction["Loss"][0]
        return count, best_score
    else:
        # if we've reached patience threshold, end training
        count += 1
        if count > patience:
            return
        # network is not training
        elif prediction["Accuracy"][0] < (starting):
            return


def train_model(
    model,
    dataset,
    epochs=15,
    device=dev,
    dtype=torch.float,
    neurogenesis=None,
    optim_fn=optim.Adam,
    optim_args={"lr": 0.0002},
    turnover=True,
    frequency=0,
    excite=False,
    end_neurogenesis=8,
    early_stop=True,
    patience=2,
    checkpoint=False,
    layer=1,
    targeted_portion=None,
    **kwargs,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim_fn(model.parameters(), **optim_args)

    log = np.zeros(epochs)
    best_score = None
    starting = None
    count = 0

    # neurogenesis
    epoch_neurogenesis = False
    batch_neurogenesis = False

    if (neurogenesis is not None) and (neurogenesis):
        if frequency:
            batch_neurogenesis = True
        else:
            epoch_neurogenesis = True

    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        if epoch >= end_neurogenesis:
            epoch_neurogenesis = False
            batch_neurogenesis = False
            model.excite = False
        for i, data in enumerate(dataset.train, 0):
            if batch_neurogenesis:
                if (epoch % frequency) == 0:
                    model.add_new(neurogenesis, turnover, targeted_portion, layer=layer)
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # If validation set exists, predict on validation set
        # Otherwise use the test set
        if dataset.valid is not None:
            prediction = predict(model, dataset, True, get_loss=False)
        elif dataset.valid is None:
            prediction = predict(model, dataset, False, get_loss=False)

        log[epoch] = prediction["Accuracy"][0]

        if epoch_neurogenesis:
            model.add_new(neurogenesis, turnover, targeted_portion, layer=layer)
            if not turnover:  # add new parameters
                optimizer.add_param_group(
                    {
                        "params": model.fc_new_in[-1].parameters(),
                        "lr": optim_args["lr"],
                        "momentum": optim_args["momentum"],
                    }
                )
                optimizer.add_param_group(
                    {
                        "params": model.fc_new_out[-1].parameters(),
                        "lr": optim_args["lr"],
                        "momentum": optim_args["momentum"],
                    }
                )

    return list(log), optimizer


def predict(model, dataset, valid=False, train=False, device=dev, get_loss=False):
    criterion = nn.CrossEntropyLoss()
    correct = []
    total = 0
    losses = []

    model.eval()
    # use the correct dataset
    if valid:
        try:
            loader = dataset.valid
        except AttributeError:
            print("No validation set. You are in test mode.")
            return
    elif train:
        loader = dataset.train
    else:
        loader = dataset.test

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # calculate loss
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # calculate accuracy (do not use softmax)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct.append((predicted == labels).sum().item())

    avg_loss = np.array(losses).mean()
    sem_loss = stats.sem(np.array(losses))

    accuracy = 100 * float(np.array(correct).sum()) / total
    sem_accuracy = 0

    #    return accuracy, avg_loss
    return {"Loss": (avg_loss, sem_loss), "Accuracy": (accuracy, sem_accuracy)}


def ablation(model, dataset, mode="random", step=0.05):
    """
    layer: layers to remove neurons
    proportion: float, fraction of neurons to ablate
    """
    assert mode in ["random", "targetted"], "mode must be random or targetted"
    model.ablate = True
    proportions = np.arange(0, 1 + step, step)
    results = np.zeros((len(proportions), 2))

    counter = 0
    for prop in proportions:
        model.ablation_prop = prop
        model.ablation_mode = mode
        acc = predict(model, dataset, train=True)
        results[counter] = (prop, acc["Accuracy"][0])
        counter += 1

    model.ablate = False
    return results


class NgnCnn(nn.Module):
    def __init__(
        self,
        layer_size=250,
        channels=3,
        control=False,
        seed=0,
        excite=False,
        neural_noise=None,
    ):
        torch.manual_seed(seed)
        super(NgnCnn, self).__init__()
        # parameters
        self.ablate = False
        self.dropout = 0
        self.channels = channels
        self.excite = excite
        self.n_new = 0
        self.control = False
        if self.control:
            self.idx_control = np.random.choice(
                range(layer_size), size=8, replace=False
            )
        self.neural_noise = neural_noise

        # 3@16x16
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.layer_size = layer_size

        self.fc_new_in = nn.ModuleList()
        self.fc_new_out = nn.ModuleList()

        if self.channels == 3:
            self.cnn_output = 64 * 4 * 4
        elif self.channels == 1:
            self.cnn_output = 64 * 9
        # three fully connected layers
        self.fcs = nn.ModuleList(
            [
                nn.Linear(self.cnn_output, self.layer_size),  # 0
                nn.Linear(self.layer_size, self.layer_size),  # 1 on dim 2 neurogenesis
                nn.Linear(self.layer_size, self.layer_size),  # 2
            ]
        )
        self.fc3 = nn.Linear(self.layer_size, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.pool4(x)

        x = x.view(-1, self.cnn_output)

        for ix, fc in enumerate(self.fcs):
            x = fc(x)
            if self.neural_noise is not None and ix == 0 and self.training:
                mean, std = self.neural_noise
                # noise = torch.log_normal(mean=mean, std=std,
                #                     size=x.size(), device=dev)
                noise = torch.zeros_like(x, device=dev)
                noise = noise.log_normal_(mean=mean, std=std)
                x = x * noise
            x = F.relu(x)

            if self.excite and ix == 1 and self.n_new and self.training:
                idx = self.idx_control if self.control else self.idx
                excite_mask = torch.ones_like(x)
                excite_mask[:, idx] = self.excite
                excite_mask.to(dev)
                x = x * excite_mask

            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = torch.renorm(x, 1, 1, 3)  # max norm

            # for ablation experiments
            if self.ablate:
                activation_size = x.size()[1]
                ablate_size = int(self.ablation_prop * activation_size)
                if self.ablation_mode == "random":
                    indices = np.random.choice(
                        range(activation_size),
                        size=size,
                        replace=False,
                    )
                if self.ablation_mode == "targetted":
                    importance = torch.norm(fc.weight.detach().clone(), p=1, dim=1)
                    indices = torch.argsort(importance, descending=True)[:ablate_size]
                x[:, indices] = 0
        x = self.fc3(x)

        return x

    def add_new(
        self,
        p_new=0.01,
        replace=True,
        targeted_portion=None,
        return_idx=False,
        layer=1,
    ):
        """
        pnew: float, proportion of hidden layer to add
        replace: float, from 0-1 which is the proportion of new neurons that replace old neurons
        target: bool, neurons that are lost are randomly chosen, or targetted
                based on variance of activity
        """
        # get a copy of current parameters
        bias = [ix.bias.detach().clone().cpu() for ix in self.fcs]
        current = [ix.weight.detach().clone().cpu() for ix in self.fcs]

        # how many neurons to add?
        if not p_new:
            return
        # if int given, use this as number of neurons to add
        if (p_new % 1) == 0:
            n_new = p_new
        # if float given, use to calculate number of neurons to add
        else:
            n_new = int(self.layer_size * p_new)

        if targeted_portion is not None:
            targ_diff = round(targeted_portion * current[layer].shape[0]) - n_new
            if targ_diff <= 0:
                n_new = n_new + targ_diff - 3

        self.n_new = n_new
        n_replace = n_new if replace else 0  # number lost
        difference = n_new - n_replace  # net addition or loss
        self.layer_size += difference  # final layer size

        # reallocate the weights and biases
        if replace:
            # if some neurons are being removed
            if targeted_portion is not None:
                try:
                    weights, mask = targeted_neurogenesis(
                        current[layer], n_replace, targeted_portion, self.training
                    )
                except ValueError:
                    print(
                        "n_replace",
                        n_replace,
                        "targ",
                        targeted_portion * (current[layer].shape[0]),
                    )

                # if neurons are targetted for removal
                idx = np.where(mask)[0]
                bias[1] = np.delete(bias[1], idx)
                current[layer] = np.delete(current[layer], idx, axis=0)
                current[layer + 1] = np.delete(current[layer + 1], idx, axis=1)
            else:
                # if neurons are randomly chosen for removal
                idx = np.random.choice(
                    range(current[layer].shape[0]), size=n_replace, replace=False
                )

                # delete idx neurons from bias and current weights (middle layer)
                bias[1] = np.delete(bias[1], idx)
                current[layer] = np.delete(current[layer], idx, axis=0)
                current[layer + 1] = np.delete(current[layer + 1], idx, axis=1)

            self.idx = idx

        # create new weight shapes
        w_in = torch.Tensor(
            self.layer_size,
            current[layer].shape[1],
        )
        b_in = torch.Tensor(self.layer_size)
        w_out = torch.Tensor(
            current[layer + 1].shape[0],
            self.layer_size,
        )

        # initialize new weights
        nn.init.kaiming_uniform_(w_in, a=math.sqrt(5))
        nn.init.kaiming_uniform_(w_out, a=math.sqrt(5))

        # in bias (out bias unaffected by neurogenesis)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w_in)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(b_in, -bound, bound)

        # put back current bias and weights into newly initiliazed layers
        b_in[:-n_new] = bias[1]
        w_in[:-n_new, :] = current[layer]
        w_out[:, :-n_new] = current[layer + 1]

        # create the parameters again
        #            self.fcs[1] = nn.Linear(current[layer].shape[1], self.layer_size)
        #            self.fcs[2] = nn.Linear(self.layer_size, current[layer+1].shape[0])
        self.fcs[layer].bias = nn.Parameter(b_in)
        self.fcs[layer].weight = nn.Parameter(w_in)
        self.fcs[layer + 1].weight = nn.Parameter(w_out)  # nn.Parameter(w_out)
        #            self.fcs[2].bias = nn.Parameter(bias[2])

        # need to send all the data to GPU again
        self.fcs.to(dev)

        if return_idx and (n_replace > 0):
            return idx
