import torch
import numpy as np
from trainer.vae import VAE
from dataset import TwitterDataset
import argparse


def beam_search(soft_words, k):
    sequences = [[list(), 0.0]]

    # walk over each step in sequence
    for tensor in soft_words:
        all_candidates = list()
        row = tensor.tolist()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda x:x[1], reverse=True)
        # select k best
        sequences = ordered[:k]

    return sequences


def main(args):

    h_dim = 64
    z_dim = 64
    c_dim = 4

    account = args.account
    n_tweets = args.n_tweets
    beam = args.beam_search
    k = args.k

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

        sample_idxs = model.sample_sentence(z, c, beam=beam, temp=0.1)

        # use beam search to find k most likely sequences
        if beam:
            seqs = beam_search(sample_idxs, k)
            print(dataset.idxs2sentence(seqs[0][0]))
        else:
            print(dataset.idxs2sentence(sample_idxs))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tweet generation')
    parser.add_argument('--gpu', default=False, type=bool, help='whether to run in the GPU')
    parser.add_argument('--n_tweets', default=100, type=int, help='number of tweets to generate')
    parser.add_argument('--account', required=True, type=str, help='account to generate tweets for')
    parser.add_argument('--beam_search', default=False, type=bool, help='whether to perform beam search or not')
    parser.add_argument('--k', required=False, type=int, help='size of beam')
    args = parser.parse_args()
    main(args)