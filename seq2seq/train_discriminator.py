import os
import torch
import yaml
import argparse
import numpy as np
from vae import VAE
import torch.optim as optim
import torch.nn.functional as F
from dataset import TwitterDataset
from torch.utils.tensorboard import SummaryWriter

def main(args):

    # hyperparams
    beta = args.beta
    lambda_c = args.lambda_c
    lambda_z = args.lambda_z
    lambda_u = args.lambda_u
    kl_weight_max = 0.4
    lr = args.lr

    epochs = args.epochs
    batch_size = args.batch_size
    device_ids = args.devices
    gpu = args.gpu
    z_dim = args.z_dim
    h_dim = args.h_dim
    report_interval = 100

    if gpu and device_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_ids

    conf = 'config.yaml'
    with open(conf) as file:
        config = yaml.safe_load(file.read())

    # load config vars
    base_dir = config.get('PROJECT_DIR')
    vocab_path = config.get('VOCAB_PATH')
    train_path = config.get('TRAIN_PATH')
    labels_path = config.get('LABELS_PATH')
    vae_path = config.get('VAE_PATH')

    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(base_dir, 'runs'))

    dataset = TwitterDataset(vocab_path=vocab_path, train_path=train_path, labels_path=labels_path, batch_size=batch_size)

    # controllable parameter for each account
    c_dim = dataset.n_accounts

    model = VAE(dataset.tweet_indexer, h_dim, z_dim, c_dim, gpu=gpu)

    device = 'cpu' if (not gpu or not torch.cuda.is_available()) else 'cuda'

    # load pre-trained generator
    model.load_state_dict(torch.load(vae_path, map_location=torch.device(device)))

    model.to(device)

    optim_G = optim.Adam(model.decoder_params, lr=lr)
    optim_D = optim.Adam(model.discriminator_params, lr=lr)
    optim_E = optim.Adam(model.encoder_params, lr=lr)

    # discriminator learning with wake-sleep algorithm
    for e in range(epochs):

        interval = 0

        # reset data loader
        dataset.resetTrainBatches()

        # generate samples and classify sentences
        for padded_inputs, idx_labels in dataset.trainIterator:

            padded_inputs = padded_inputs.to(device)
            idx_labels = idx_labels.to(device)

            """update params of discriminator, sleep phase"""
            x_gen, c_gen = model.generate_sentences(batch_size)
            c_hat = torch.argmax(c_gen, dim=1)

            # classify twitter handle of sentences
            y_hat_real = model.forwardDiscriminator(padded_inputs)

            if gpu:
                x_gen = x_gen.cuda()

            y_hat_gen = model.forwardDiscriminator(x_gen)

            # entropy used to obtain high confidence in predictions
            entropy = F.log_softmax(y_hat_gen, dim=1).mean()
            entropy = -entropy

            # supervised loss for semantic meaning
            loss_s = F.cross_entropy(y_hat_real, idx_labels.squeeze(1))

            # unsupervised loss
            loss_u = F.cross_entropy(y_hat_gen, c_hat)

            loss_d = loss_s + lambda_u * (loss_u + beta * entropy)

            writer.add_scalar('Discriminator/discriminator', loss_d, e)

            loss_d.backward()
            optim_D.step()
            optim_D.zero_grad()

            """ update params of generator, sleep phase """
            model.train()

            input_lens = torch.tensor(np.count_nonzero(padded_inputs.cpu(), axis=1))
            recon_loss, kl_loss = model(padded_inputs, input_lens, use_c_prior=False)

            x_gen_attr, target_z, target_c = model.generate_soft_embed(batch_size)

            """
            Feed soft generated sentence to discriminator
            to measure fitness to the target attribute.
            """
            x_gen_len = torch.tensor([x_gen_attr.shape[1]]*batch_size)
            y_z, *_ = model.forwardEncoderEmb(x_gen_attr, x_gen_len)
            y_c = model.forwardDiscEmbed(x_gen_attr)

            loss_vae = recon_loss + kl_weight_max * kl_loss
            loss_attr_c = F.cross_entropy(y_c, target_c)
            loss_attr_z = F.mse_loss(y_z, target_z)

            loss_g = loss_vae + lambda_c*loss_attr_c + lambda_z*loss_attr_z
            # writer.add_scalar('Discriminator/generator', loss_g, e)

            loss_g.backward()

            optim_G.step()
            optim_G.zero_grad()

            """ update params of encoder, wake phase """
            recon_loss, kl_loss = model(padded_inputs, input_lens, use_c_prior=False)

            loss_e = recon_loss + kl_weight_max * kl_loss

            # writer.add_scalar('Discriminator/encoder', loss_e, e)

            loss_e.backward()

            optim_E.step()
            optim_E.zero_grad()

            if interval % report_interval == 0:
                z = model.sample_z_prior(1)
                c = model.sample_c_prior(1)

                sample_idxs = model.sample_sentence(z, c)
                sample_sent = dataset.idxs2sentence(sample_idxs)

                print(f'Epoch-{e}; loss_D: {loss_d.item():.4f}; loss_G: {loss_g.item():.4f}')

                _, c_idx = torch.max(c, dim=1)

                print(f'c = {dataset.idx2label(int(c_idx))}')
                print(f'Sample: {sample_sent}')
                writer.add_text('Generator', sample_sent, e)

            interval += 1

    # save model parameters
    saveModel(model, base_dir)
    writer.flush()
    writer.close()

def saveModel(model, base_dir):

    models_dir = os.path.join(base_dir, 'models')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    PATH = os.path.join(models_dir, 'tweet_gen.pt')

    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--batch_size', default=32, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--lambda_c', default=0.1, type=float)
    parser.add_argument('--lambda_z', default=0.1, type=float)
    parser.add_argument('--lambda_u', default=0.1, type=float)
    parser.add_argument('--h_dim', default=64, type=int, help='Dimensionality of hidden state')
    parser.add_argument('--z_dim', default=64, type=int, help='Dimensionality of latent space')
    parser.add_argument('--gpu', default=False, type=bool, help='Flag to run model on gpu')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--devices', required=False, type=str, help='Device ids to train model on')

    args = parser.parse_args()

    main(args)