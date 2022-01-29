"""
Author: Lina Tran
Date: March 22, 2018
"""

import torch
from torch.nn import Parameter
import torch.nn as nn
import numpy as np
import random
import copy
from mnist import MNIST
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.cifar import CIFAR10
from neurodl.targeted_neurogenesis import targeted_neurogenesis
import os


class EMNIST(Dataset):
    """Convert MNIST data folder to PyTorch Dataset object."""

    def __init__(self, folder, train, transform=None):
        """
        Args:
            - folder (string): Directory with all the data.
            - train (bool): Load training test, if False, load testing.
            - transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder = folder
        self.transform = transform
        self.data = MNIST(folder)
        self.data.select_emnist("mnist")

        if train:
            images, labels = self.data.load_training()
        else:
            images, labels = self.data.load_testing()

        self.labels = labels

        self.images = np.array(images).reshape((len(images), 28, 28)).astype(np.float32)

    def __getitem__(self, idx):
        sample = np.array(self.images[idx])
        if type(idx) == int:
            sample = np.expand_dims(sample, axis=2)

        if self.transform:
            sample = self.transform(sample)

        return sample, self.labels[idx]

    def __len__(self):
        return len(self.images)


def load_data(batch_size, data_dir, fashion=False, c10=False, num_workers=16, split=0.85):
    """
    Load pytorch datasets for mnist and emnist.
    ===
    Args:
        - batch_size (int): size of batches for dataloader
        - data_dir (str): path to the data file
            default = False
        - split (float): how to split the training dataset for validation
    Output:
        - (fashion) mnist pytorch dataset
        of three DataLoaders (train, validation and test loaders)
    """
    print("Loading training and test data.")
    # MNIST, train & test
    if fashion:
        mnist = [
            datasets.FashionMNIST(
                data_dir, train=True,
                transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )),
            datasets.FashionMNIST(
                data_dir, train=False,
                transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )),
        ]
    elif c10:
        mnist = [
            datasets.CIFAR10(
                data_dir, train=True,
                transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )),
            datasets.CIFAR10(
                data_dir, train=False,
                transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )),
        ]

    else:
        mnist = [
            EMNIST(
                data_dir,
                train=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
                ),
            ),
            EMNIST(
                data_dir,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
                ),
            ),
        ]

    from sklearn.model_selection import train_test_split

    # split train/validation
    num_train = len(mnist[0])
    indices = list(range(num_train))
    print(num_train, indices[-1])

    X_train, X_valid = train_test_split(
        indices, test_size=int(num_train * (1 - split)), random_state=0
    )

    train_sampler = SubsetRandomSampler(X_train)
    valid_sampler = SubsetRandomSampler(X_valid)

    train_mnist = torch.utils.data.DataLoader(
        mnist[0],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    valid_mnist = torch.utils.data.DataLoader(
        mnist[0],
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    full_train_mnist = torch.utils.data.DataLoader(
        mnist[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_mnist = torch.utils.data.DataLoader(
        mnist[1], batch_size=batch_size, num_workers=num_workers, drop_last=True
    )

    mnist = [train_mnist, valid_mnist, full_train_mnist, test_mnist]

    print("MNIST loaded.")
    return mnist


def predict(net, criterion, test=True, train=False):
    """
    Given a model and criterion, get prediction performance on
    validation or test set.
    ===
    args:
        - net: pytorch model object, model to test
        - criterion: function, loss function
        - test: bool, if set to True, uses Test set, otherwise Validation set,
            default=True
        - combine: bool, add Validation to Test Set, negated if test=False,
            default=False
    output:
        - prediction accuracy on test set
    """
    net.eval()
    correct_cnt, avg_loss = 0, 0
    if test:
        loader = net.test_mnist
    elif train:
        loader = net.train_mnist
    else:
        loader = net.valid_mnist
    for batch_idx, (x, target) in enumerate(loader):
        x, target = x.to(net.device), target.to(net.device)
        output = net(x)
        loss = criterion(output, target)
        _, pred_label = torch.max(output.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        avg_loss += loss.item()

    total = len(loader) * loader.batch_size
    avg_loss /= len(loader)
    accuracy = float(correct_cnt) / total

    return avg_loss, accuracy


def ablation(model, criterion, layer, repeats=5, step=0.05):
    """
    layer: list, layers to remove neurons
    proportion: float, fraction of neurons to ablate
    """
    model.ablation = True
    model.ablation_layer = layer
    proportions = np.arange(0, 1 + step, step)
    results = np.zeros((repeats * len(proportions), 2))

    counter = 0
    for repeat in range(repeats):
        for prop in proportions:
            model.ablation_prop = prop
            loss, acc = predict(model, criterion, test=False)
            results[counter] = (prop, acc)
            counter += 1

    model.ablation = False
    return results


def train_model(
    model,
    opt_fn,
    opt_args,
    criterion,
    epochs=40,
    replace=True,
    full_set=False,
    patience=5,
    freq=500,
    end_neurogenesis=40,
    neurogenesis=False,
    new_args=None,
    early_stop=False,
    **kwargs
):
    """
    model: object, the NGNModel to be trained
    opt: optimizer
    criterion: loss function
    epochs: int > 0, number of epochs to train model
    full_set: bool, use full training set, else use split and validation.
    patience: int > 0, patience for early stopping, default=5
    neurogenesis: bool, excite weights requires new neurons added
        every epoch/batch, default=False
    new_args: dict, arguments for model.add_new() if neurogenesis,
        default=None
    """
    opt = opt_fn(model.parameters(), **opt_args)
    log = np.zeros(epochs)

    if full_set:
        loader = model.full_train_mnist
    else:
        loader = model.train_mnist
    for epoch in range(epochs):
        model.train()
        if epoch >= end_neurogenesis:
            neurogenesis = False
            model.excite=False
        for batch_idx, (x, target) in enumerate(loader):
            x, target = x.to(model.device), target.to(model.device)
            # zero the gradients, if they exist
            opt.zero_grad()
            # forward pass
            output = model(x)
            loss = criterion(output, target)
            # backward pass
            loss.backward()
            opt.step()
            # excite every epoch
            if neurogenesis and (batch_idx % freq == 0):
                model.add_new(**new_args)
                if not new_args['replace']:
                    opt = opt_fn(model.parameters(), **opt_args)
                model.to(model.device)


        avg_loss, accuracy = predict(model, criterion, full_set)
        log[epoch] = accuracy


    return list(log)


def ablate_forward(x, proportion):
    activation_size = x.size()[1]
    indices = np.random.choice(
        range(activation_size), size=int(proportion * activation_size), replace=False
    )
    x[:, indices] = 0
    return x


def excite_forward(x, increment, proportion, n_new, batch_size, device):
    excite = int(proportion * n_new) - 1
    # chooses indices from new neurons to randomly activate
    indices = torch.from_numpy(
        np.random.rand(batch_size, n_new).argpartition(excite, axis=1)[
            :, :excite
        ]
        + increment
    ).to(device)

    activity = torch.ones(indices.size(), device=device)
    x = x.scatter(1, indices, activity)
    return x


def target_replacement(model, layer, mode="variance"):
    #    loader = model.train_mnist
    activations, targets = model.return_layer(layer=layer, dataset="train")
    # checks activations per class
    return activations


class NgnMlp(nn.Module):
    """
    Model that can take input of mnist/emnist handwritten digit/letter images,
    for the purpose of studying impact of new neuron addition, or turnover.
    """

    def __init__(self, dataset, layer_size=[250,250,250], dropout=0,
                 control=False, excite=False, excite_val=1.3,
                 eval_mode=False, c10=False, neural_noise=None):
        super(NgnMlp, self).__init__()
        self.relu = nn.ReLU()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.NLLLoss  # loss criterion fn
        self.layer_size = layer_size
        self.control = control
        if self.control:
            self.modified_layers = [1]
            self.idx_control = np.random.choice(np.arange(self.layer_size[1]), size=int(self.layer_size[1]*0.05), replace=False)
        self.neural_noise = neural_noise

        # parameters
        self.new = 0  # number of new units
        self.excite = excite  # excitability bool
        self.excite_val = excite_val  # excitability proportion
        self.dropout = dropout  # dropout proportion
        self.lr_new = 1  # learning rate multiplier for new units
        self.eval_mode = eval_mode  # evaluate model using test set
        self.modified_layers = []  # which layers were modified
        self.ablation = False  # whether or not we are doing ablation expt
        self.ablation_layer = []  # which layers to ablate

        # define datasets
        self.train_mnist = dataset[0]
        self.valid_mnist = dataset[1]
        self.full_train_mnist = dataset[2]
        self.test_mnist = dataset[3]

        self.batch_size = self.train_mnist.batch_size

        if c10:
            self.input_size = 32*32*3
        else:
            self.input_size = 784
        self.output_size = 10

        # initialize weights
        self.fcs = nn.ModuleList([nn.Linear(self.input_size, self.layer_size[0])])

        self.fcs.extend(
            [nn.Linear(
                self.layer_size[ix], self.layer_size[ix+1]
                ) for ix, i in enumerate(layer_size[1:])]
        )

        self.fcs.append(nn.Linear(self.layer_size[-1], self.output_size, bias=False))

        weights_shape = self.fcs[1].weight.shape[0]

    def forward(self, x, return_layer=None):
        full_layer = None
        x = x.view(self.batch_size, self.input_size)
        for ix, fc in enumerate(self.fcs[:-1]):
            x = fc(x)
            # neural noise
            if self.neural_noise is not None and ix==0 and self.training:
                mean, std = self.neural_noise
                noise = torch.zeros_like(x, device=self.device)
                noise = noise.log_normal_(mean=mean, std=std)
#                noise = torch.normal(mean=mean, std=std,
#                                     size=x.size())
                x *= noise
            x = self.relu(x)
            # if ablation expt
            if (ix in self.ablation_layer) and self.ablation:
                x = ablate_forward(x, self.ablation_prop)

            # dropout
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = torch.renorm(x, 1, 1, 3) # max norm

            # if we've added new neurons + excite is True
            if ix in self.modified_layers or self.excite and self.training:
                #increment = self.dg_size - self.new
                idx = self.idx_control if self.control else self.idx
                # if eval_mode is True, do no excite during forward pass if testing
                excite_mask = torch.ones_like(x)
                excite_mask[:, idx] = self.excite_val
                excite_mask.to(self.device)
                x = x * excite_mask

            # return a layer's activations
            if ix == return_layer:
                full_layer = x.clone()
        x = F.log_softmax(self.fcs[-1](x), dim=1)

        if return_layer is not None:
            return x, full_layer
        else:
            return x

    def add_new(
        self, pnew=0.01, layers=[1], replace=True, targeted_portion=None,
    ):
        """
        Neurons replaced randomly, or via targeted strategy. The new
        neurons are always added to the end of the updated array.
        pnew: float, proportion of hidden layer to add
        excite: float, proportion of new to excite
        layers: list, which hidden layers to add new neurons to
        replace: float, proportion of new neurons that replace old neurons

        """
        if pnew == 0:
            return

        # parameter updates
        self.dg_size = self.layer_size[layers[0]]
        if (pnew % 1) == 0:
            n_new = pnew
        else:
            n_new = int(self.dg_size * pnew)


        # add new neurons to hidden layer
        # get current weights as float tensors
        self.modified_layers = layers

        # in the case of replacement instead of additive
        if replace:
            p_replace = n_new
            if p_replace > self.dg_size:
                p_replace = self.dg_size
        else:
            self.dg_size += n_new
            p_replace = 0

        for i in layers:
            assert len(layers) < len(
                self.fcs
            ), "# of layers {} must be fewer than # of weights {}".format(
                len(layers), len(self.fcs)
            )

            # save current weights
            bias = [ix.bias.clone().detach().cpu() for ix in self.fcs[:-1]]
            current = [ix.weight.clone().detach().cpu() for ix in self.fcs]
            # initialize new weights
            import math
            hl_input = torch.Tensor(self.dg_size, current[i].shape[1])
            nn.init.kaiming_uniform_(hl_input, nonlinearity="relu")
            #nn.init.xavier_uniform_(hl_input, gain=nn.init.calculate_gain("relu"))
            #hl_input = hl_input.clamp(min=-1, max=1)

            hl_output = torch.zeros(current[i + 1].shape[0], self.dg_size)
            nn.init.kaiming_uniform_(hl_output, nonlinearity="relu")
            #nn.init.xavier_uniform_(hl_output, gain=nn.init.calculate_gain("relu"))
            #hl_output = hl_output.clamp(min=-1, max=1)

            # if replacement, delete some neurons to replace with new neurons
            if p_replace > 0:
                if targeted_portion is not None:
                    try:
                        weights, mask = targeted_neurogenesis(
                            current[i], p_replace, targeted_portion,
                            self.training)
                    except ValueError:
                        p_replace = targeted_portion * current[i].shape[0] - 2

                    # if neurons are targetted for removal
                    idx = np.where(mask)[0]
                elif not self.control:
                    idx = np.arange(current[i].shape[0])[-p_replace:]
                else:
                    idx = np.random.choice(
                        range(current[i].shape[0]),
                        size=p_replace,
                        replace=False
                    )
                    self.idx_control = idx
                self.idx = idx
                current[i] = np.delete(current[i], idx, axis=0)
                current[i + 1] = np.delete(current[i + 1], idx, axis=1)
                bias[i] = np.delete(bias[i], idx)

            # concatenate old and new neurons
            #new_wi = torch.cat([current[i], hl_input], dim=0)
            #new_wo = torch.cat([current[i + 1], hl_output], dim=1)

            # put back current bias and weights into newly initialized layers
            hl_input[:-n_new, :] = current[i]
            hl_output[:, :-n_new] = current[i+1]

            import math

            new_bias = torch.Tensor(self.dg_size)
            # in bias only (out bias unaffected by neurogenesis)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(hl_input)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_bias, -bound, bound)

            if replace:
                new_bias[:-n_new] = bias[i]
                self.fcs[i].bias.data = nn.Parameter(new_bias)

                # reallocate the weights and biases
                self.fcs[i].weight = nn.Parameter(hl_input)
                self.fcs[i + 1].weight = nn.Parameter(hl_output)

                # if last hidden layer, no bias
                try:
                    self.fcs[i + 1].bias.data = bias[i + 1]
                except IndexError:
                    pass
            else:
                new_bias[:-n_new] = bias[i]
                self.fcs[i].bias = torch.nn.Parameter(new_bias)

                # reallocate the weights and biases
                self.fcs[i].weight = nn.Parameter(hl_input)
                self.fcs[i + 1].weight = nn.Parameter(hl_output)

                # if last hidden layer, no bias
                try:
                    self.fcs[i + 1].bias.data = bias[i + 1]
                except IndexError:
                    pass

        self.new = n_new
        self.layer_size[layers[0]] = self.dg_size

    def return_layer(self, layer, dataset="Train"):
        """
        layer: int, layer index to return
        dataset: str, one of Train, Validation or Test
        """
        assert dataset in [
            "Train",
            "Valid",
            "Test",
        ], "dataset must be one of Train, Valid, or Test"
        if dataset == "Train":
            dataset = self.train_mnist
        elif dataset == "Valid":
            dataset = self.valid_mnist
        else:
            dataset = self.test_mnist

        full_layer = torch.zeros(len(dataset.dataset), self.layer_size[layer])
        targets = torch.zeros(len(dataset.dataset), 1)

        for batch_idx, (x, target) in enumerate(dataset):
            x, target = x.to(self.device), target.to(self.device)
            self.eval()
            output, activities = self.forward(x, layer)
            full_layer[
                batch_idx * dataset.batch_size : batch_idx * dataset.batch_size
                + dataset.batch_size
            ] = activities
            targets[
                batch_idx * dataset.batch_size : batch_idx * dataset.batch_size
                + dataset.batch_size
            ] = target.view(dataset.batch_size, 1)

        return full_layer, targets

    def get_weights(self, wm_idx):
        assert wm_idx < len(
            self.fcs
        ), "wm_idx must be less than number of weight matrices, {}".format(
            len(self.fcs)
        )
        self.fcs[layer]
