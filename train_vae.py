import os
import math
import torch
import argparse
import numpy as np
import torch.optim as optim
from vae import VAE

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=.001)
parser.add_argument('--gpu', default=False, help='Flag to run model on gpu')
parser.add_argument('--epochs', default=1000, help='Training epochs')

args = parser.parse_args()\



def main(args):

    lr = args.lr
    epochs = args.epochs
    gpu = args.gpu
    batch_size = 32
    z_dim = 20
    h_dim = 64
    lr_decay_every = 1000000
    log_interval = 1000
    z_dim = h_dim

    # number of controllable params
    c_dim = 2

    model = VAE()
