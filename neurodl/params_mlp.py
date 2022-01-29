import numpy as np
import torch.optim as optim
import torch

OPTIMIZER =  optim.SGD
EPOCHS =  35
DROPOUT_EPOCHS =85 
BATCH_SIZE = 100
LR = 0.05
DTYPE = torch.float
FREQUENCY=200
PNEW = 0.02
NEURAL_NOISE = (-0.5, 0.5)