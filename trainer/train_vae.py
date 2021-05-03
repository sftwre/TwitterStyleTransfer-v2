import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from dataset import TwitterDataset
from vae import VAE


def main(args):

    # tensorboard writer
    log_runs = args.log
    lr = args.lr
    epochs = args.epochs
    gpu = args.gpu
    z_dim = args.z_dim
    h_dim = args.h_dim
    batch_sz = args.batch_size
    device_ids = args.devices
    lr_decay_every = 5
    report_interval = 100

    # mask available devices
    if gpu and device_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids

    if log_runs:
        writer = SummaryWriter()

    dataset = TwitterDataset(batch_size=batch_sz)

    # controllable parameter for each account
    c_dim = dataset.n_accounts

    model = VAE(dataset.tweet_indexer, h_dim, z_dim, c_dim, gpu=gpu)

    # Annealing for KL term
    kld_start_inc = 3000
    kld_weight = 0.01
    kld_max = 0.15
    kld_inc = (kld_max - kld_weight) / (epochs - kld_start_inc)

    # optimization algorithm
    optimizer = optim.Adam(model.vae_params, lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_devices = torch.cuda.device_count()

    # parallelize training if possible
    if n_devices > 1:
        device_ids = [i for i in range(n_devices)]
        model = nn.DataParallel(model, device_ids)

    model.to(device)

    # for each epoch, train on data
    for e in range(epochs):

        interval = 0
        dataset.resetTrainBatches()

        for padded_inputs, padded_labels in dataset.trainIterator:

            # zero out previous gradients
            optimizer.zero_grad()

            input_lens = torch.tensor(np.count_nonzero(padded_inputs, axis=1))

            # cast to correct device
            padded_inputs = padded_inputs.to(device)

            recon_loss, kl_loss = model.forward(padded_inputs, input_lens)

            loss = recon_loss + kld_weight * kl_loss

            # Anneal kl_weight
            if e > kld_start_inc and kld_weight < kld_max:
                kld_weight += kld_inc

            # sum loss across multiple GPU's
            if isinstance(model, nn.DataParallel):
                loss.sum().backward()
            else:
                loss.backward()

            # grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
            optimizer.step()

            recon_loss = recon_loss.sum().item()
            kl_loss = kl_loss.sum().item()
            loss = loss.sum().item()

            if log_runs:
                writer.add_scalar('VAE/recon_loss', recon_loss, e)
                writer.add_scalar('VAE/kl_loss', kl_loss, e)
                writer.add_scalar('VAE/loss', loss, e)


            if interval % report_interval == 0:

                if isinstance(model, nn.DataParallel):
                    z = model.module.sample_z_prior(1)
                    c = model.module.sample_c_prior(1)
                    sample_idxs = model.module.sample_sentence(z, c)
                else:
                    z = model.sample_z_prior(1)
                    c = model.sample_c_prior(1)
                    sample_idxs = model.sample_sentence(z, c)

                sample_sent = dataset.idxs2sentence(sample_idxs)

                print(f'Epoch-{e}; Loss: {loss:.4f}; Recon: {recon_loss:.4f}; KL: {kl_loss:.4f}')
                print(f'Sample: "{sample_sent}"', end='\n')

                if log_runs:
                    writer.add_text('Generator - training', sample_sent, e)

            # Anneal learning rate
            new_lr = lr * (0.5 ** (e // lr_decay_every))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            interval += 1

    # detach from gpu's
    if isinstance(model, nn.DataParallel):
        model = model.module

    saveModel(model)

    if log_runs:
        writer.flush()
        writer.close()


def saveModel(model):

    if not os.path.exists('../models/'):
        os.makedirs('../models/')

    PATH = '../models/vae.pt'

    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--log', dest='log', action='store_true', help='Flag to log training loss/params for tensorboard visualizations')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='Number of training samples per iteration')
    parser.add_argument('--no-log', dest='log', action='store_false', help='Flag to turn off tensorboard logging')
    parser.set_defaults(log=True)

    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Flag to run model on gpu')
    parser.add_argument('--cpu', dest='gpu', action='store_false', help='Flag to run model on cpu')
    parser.set_defaults(gpu=True)

    parser.add_argument('--devices', required=False, type=str, help='Device ids to train model on')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--h_dim', default=64, type=int, help='Dimensionality of hidden state')
    parser.add_argument('--z_dim', default=64, type=int, help='Dimensionality of latent space')

    args = parser.parse_args()

    main(args)