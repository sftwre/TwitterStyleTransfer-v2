import os
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from dataset import TwitterDataset
from vae import VAE


def main(args):

    # tensorboard writer
    writer = SummaryWriter()
    log_runs = args.log
    lr = args.lr
    epochs = args.epochs
    gpu = args.gpu
    z_dim = args.z_dim
    h_dim = args.h_dim
    lr_decay_every = 1000000
    report_interval = 100

    dataset = TwitterDataset(gpu=gpu)

    # controllable parameter for each account
    c_dim = dataset.n_accounts

    model = VAE(dataset.vocab_size, h_dim, z_dim, c_dim, gpu=gpu)

    # Annealing for KL term
    kld_start_inc = 3000
    kld_weight = 0.01
    kld_max = 0.15
    kld_inc = (kld_max - kld_weight) / (epochs - kld_start_inc)

    # optimization algorithm
    optimizer = optim.Adam(model.vae_params, lr=lr)

    try:

        # for each epoch, train on data
        for e in range(epochs):

            interval = 0
            dataset.resetTrainBatches()

            for inputs, labels in dataset.trainIterator:
                recon_loss, kl_loss = model.forward(inputs)

                if log_runs:
                    writer.add_scalar('VAE/recon_loss', recon_loss, e)
                    writer.add_scalar('VAE/kl_loss', kl_loss, e)
                    writer.add_scalar('VAE/loss', loss, e)

                loss = recon_loss + kld_weight * kl_loss

                # Anneal kl_weight
                if e > kld_start_inc and kld_weight < kld_max:
                    kld_weight += kld_inc

                loss.backward()
                # grad_norm = torch.nn.utils.clip_grad_norm(model.vae_params, 5)
                optimizer.step()
                optimizer.zero_grad()

                if interval % report_interval == 0:
                    z = model.sample_z_prior(1)
                    c = model.sample_c_prior(1)

                    sample_idxs = model.sample_sentence(z, c)
                    sample_sent = dataset.idxs2sentence(sample_idxs)

                    print(f'Epoch-{e}; Loss: {loss.item():.4f}; Recon: {recon_loss.item():.4f}; KL: {kl_loss.item():.4f}')
                    print(f'Sample: "{sample_sent}"', end='\n')

                # Anneal learning rate
                new_lr = lr * (0.5 ** (e // lr_decay_every))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                interval += 1
        saveModel(model)
        writer.flush()
        writer.close()

    except KeyboardInterrupt:
        saveModel(model)


def saveModel(model):

    if not os.path.exists('models/'):
        os.makedirs('models/')

    PATH = 'models/vae.pt'

    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--log', default=False, type=str, help='Flag to log training loss/params for tensorboard visualizations')
    parser.add_argument('--gpu', default=False, type=bool, help='Flag to run model on gpu')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--h_dim', default=64, type=int, help='Dimensionality of hidden state')
    parser.add_argument('--z_dim', default=64, type=int, help='Dimensionality of latent space')

    args = parser.parse_args()

    main(args)