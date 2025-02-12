import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import pdb
import random
from torch.backends import cudnn
from experiments.anticipation.anticipation_opts import parser
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from utils import read_mapping_dict
from data.common.baseanticipationdataset import BaseDataset
from models.action_anticipation.futr import FUTR
from trainers.anticipation_trainer import train
from trainers.anticipation_trainer import predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actions_dict={'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7}

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available. Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU.")

def main():
    args= parser.parse_args()
