from torch import optim
import math
import torch
from torch.nn import Parameter
import torch.nn as nn
import neurodl.cnn as cnn
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


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adaptive_neurogenesis(scaling_factor, current_neurogenesis, batch, c=.98):
    """
    Given a matrix of average weight changes from the previous epoch, 
    take the l2 norm, generate new neurogenesis parameters based on log of the
    weight changes, given a constant, c.
    updated_neurogenesis = C(log((deltas)^2))

    Args:
        square_delta_weights (torch.tensor): NxM matrix of squared average weight 
            changes from the previous epoch of batch updates.
        current_neurogenesis (int): Current number of neurons added per episode
        c (float): constant to scale neurogenesis rate
    Returns:
        adapted_neurogenesis: int, new neurogenesis value for model 
    """
    print(scaling_factor)
    adapted_neurogenesis = np.round(scaling_factor * current_neurogenesis * c)
    if (batch % 1) == 0: #TODO
        print("new neurogenesis rate:", adapted_neurogenesis)
    return int(adapted_neurogenesis)


class Adam_mod(optim.Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(Adam_mod, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_mod, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None, get_lrs=[]):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        adaptive_data = {
            "step_sizes": {},
            "factor": {},
        }

        for group in self.param_groups:
            for ip, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if ip in get_lrs:
                    adaptive_data["step_sizes"][ip] = step_size
                    adaptive_data["factor"][ip] = (
                        math.sqrt(bias_correction2) / bias_correction1
                    )

        if get_lrs is not None:
            return adaptive_data
        return loss


def train_model_adaptive(
    model,
    dataset,
    epochs=15,
    device=dev,
    dtype=torch.float,
    neurogenesis=None,
    optim_fn=Adam_mod,
    optim_args={},
    adaptive=False,
    turnover=True,
    end_neurogenesis=10,
    frequency=600,
    early_stop=True,
    patience=2,
    checkpoint=False,
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

    get_lrs = [14]

    if (neurogenesis is not None) and (neurogenesis):
        if frequency:
            batch_neurogenesis = True
        else:
            epoch_neurogenesis = True

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_avg_new = np.zeros_like(model.fcs[1])
        if epoch >= end_neurogenesis:
            epoch_neurogenesis = False
            batch_neurogenesis = False
        if adaptive and epoch_neurogenesis and epoch > 0:
            neurogenesis = adaptive_neurogenesis(
                adam_params["factor"][14], neurogenesis, i
            )

        for i, data in enumerate(dataset.train, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            adam_params = optimizer.step(get_lrs=get_lrs)

            if batch_neurogenesis and ((i % frequency) == 0):
                if adaptive and i > 0:
                    neurogenesis = adaptive_neurogenesis(
                        adam_params["factor"][14], neurogenesis, i,
                    )
                    print(neurogenesis)

                if (i % frequency) == 0:
                    model.add_new(neurogenesis, turnover, targeted_portion)

        if epoch_neurogenesis and (epoch < (epochs - 1)):
            model.add_new(neurogenesis, turnover, targeted_portion)
            optimizer = optim_fn(model.parameters(), **optim_args)

        # If validation set exists, predict on validation set
        # Otherwise use the test set
        if dataset.valid is not None:
            prediction = cnn.predict(model, dataset, True, get_loss=False)
        elif dataset.valid is None:
            prediction = cnn.predict(model, dataset, False, get_loss=False)

        log[epoch] = prediction["Accuracy"][0]

        running_avg_new += adam_params["step_sizes"][14]

    return list(log), adam_params
