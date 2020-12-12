import torch
import numpy as np
from vae import VAE
from dataset import TwitterDataset
import argparse


def main(args):

    h_dim = 64
    z_dim = 64
    c_dim = 4

    account = args.account
    n_tweets = args.n_tweets

    accounts = ['elon', 'dril', 'donald', 'dalai']

    device = 'cuda:0' if args.gpu else 'cpu'

    model_path = './models/tweet_gen.pt'

    dataset = TwitterDataset(gpu=args.gpu)

    model = VAE(dataset.vocab_size, h_dim, z_dim, c_dim, gpu=args.gpu)

    model.load_state_dict(torch.load(model_path))

    # load model on correct device
    model.to(device)

    dalai = np.array([1, 0, 0, 0]).reshape(1, -1)
    donald = np.array([0, 1, 0, 0]).reshape(1, -1)
    elon = np.array([0, 0, 1, 0]).reshape(1, -1)
    dril = np.array([0, 0, 0, 1]).reshape(1, -1)

    # get code for account
    if account not in accounts:
        print(f'{account} not supported yet :(')
        exit(-1)

    if account == 'donald':
        code = donald
    elif account == 'dalai':
        code = dalai
    elif account == 'dril':
        code = dril
    else:
        code = elon

    # generate tweets for account
    c = torch.FloatTensor(code).to(device)
    _, c_idx = torch.max(c, dim=1)

    print(f'Twitter account: @{dataset.idx2label(int(c_idx))}')

    for _ in range(n_tweets):
        # Samples latent and conditional codes randomly from prior
        z = model.sample_z_prior(1)
        sample_idxs = model.sample_sentence(z, c, temp=0.1)
        print(dataset.idxs2sentence(sample_idxs))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tweet generation')
    parser.add_argument('--gpu', default=False, type=bool, help='whether to run in the GPU')
    parser.add_argument('--n_tweets', default=100, type=int, help='number of tweets to generate')
    parser.add_argument('--account', required=True, type=str, help='account to generate tweets for')
    args = parser.parse_args()
    main(args)