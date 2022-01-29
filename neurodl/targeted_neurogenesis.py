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


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


def targeted_neurogenesis(weights, n_replace, targeted_portion, is_training):
    """
    Takes a weight matrix and applied targetted dropout based on weight
    importance (From Gomez et al. 2019; https://for.ai/blog/targeted-dropout/)

    Args:
        weights - the input by ouput matrix of weights
        dropout_rate - float (0,1), the proprotion of targeted neurons to dropout
        targeted_portion - the proportion of neurons/weights to consider 'unimportant'
            from which dropout_rate targets from
        is_training - bool, whether model is training, or being evaluated
    """
    # get the input vs output size
    weights_shape = weights.shape

    # l1-norm of neurons based on input weights to sort by importance
    importance = torch.norm(weights, p=1, dim=1)

    # chose number of indices to remove of the output neurons
    idx = round(targeted_portion * weights_shape[0]) - 1

    # when sorting the abs valued weights ascending order
    # take the index of the targeted portion to get a threshold
    importance_threshold = torch.sort(importance)[0][-idx] # TODO -idx

    # only weights below threshold will be set to None
    unimportance_mask = importance < importance_threshold  #TODO > change < regular

    # during evaluation, only use important weights, without dropout threshold
    if not is_training:
       weights = torch.reshape(weights, weights_shape)
       return weights

    # difference between dropout_rate and unimportance_mask (i.e. threshold)
    idx_drop = np.random.choice(np.where(unimportance_mask)[0], size=n_replace, replace=False)
    dropout_mask = torch.zeros_like(unimportance_mask)
    dropout_mask[idx_drop] = 1
    
    # delete dropped out units
    weights = weights[~dropout_mask]

    return weights, dropout_mask


def adaptive_neurogenesis(delta_weights, current_neurogenesis, moment):
    """
    Given a matrix of average weight changes from the previous epoch,
    generate new neurogenesis parameters, given a constant, c.
    
        Args:
        delta_weights (torch.tensor): NxM matrix of average weight changes
            from the previous epoch of batch updates.  
        current_neurogenesis (int): Current number of neurons added per episode 
        c ([type]): constant to scale neurogenesis rate
    """
    
