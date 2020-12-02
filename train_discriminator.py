import torch
import argparse
from vae import VAE
import torch.optim as optim
import torch.functional as F
from dataset import TwitterDataset


def main(args):

    lr = args.lr
    epochs = args.epochs
    gpu = args.gpu
    z_dim = 64
    h_dim = 64
    lr_decay_every = 1000000
    report_interval = 100

    dataset = TwitterDataset(gpu=gpu)

    # controllable parameter for each account
    c_dim = dataset.n_accounts

    model = VAE(dataset.vocab_size, h_dim, z_dim, c_dim, gpu=gpu)

    device = 'cpu' if not gpu else 'cuda:0'

    # load pre-trained generator
    model.load_state_dict(torch.load('models/vae.pt', map_location=torch.device(device)))

    optim_G = optim.Adam(model.decoder_params, lr=lr)
    optim_D = optim.Adam(model.discriminator_params, lr=lr)
    optim_E = optim.Adam(model.encoder_params, lr=lr)


    for e in range(epochs):
        dataset.resetTrainBatches()

        for inputs, labels in dataset.trainIterator:

            bs = inputs.size(0)

            # updated discriminator
            X_gen, c_gen = model.generate_sentences(bs)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--gpu', default=False, type=bool, help='Flag to run model on gpu')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')

    args = parser.parse_args()

    main(args)