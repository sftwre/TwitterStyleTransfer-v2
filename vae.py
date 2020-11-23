import torch
import numpy as np
import torch.nn as nn
import torch.functional as F
from itertools import chain

class VAE(nn.Module):

    def __init__(self, vocab_size, h_dim=64, z_dim=20, c_dim=2, tweet_len=280):
        super(VAE, self).__init__()


        self.emb_dim = h_dim
        self.z_dim = z_dim
        self.vocab_size = vocab_size
        self.embedder = nn.Embedding(vocab_size, h_dim)

        """
        encoder is uni-directional LSTM with fully connected 
        layer on top of last hidden unit to model the mean
        and log variance of latent space.
        """
        self.encoder = nn.LSTM(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)

        """
        decoder is LSTM with embeddings, z, and c as inputs
        """
        self.decoder = nn.LSTM(self.emb_dim+z_dim+c_dim, z_dim+c_dim, dropout=0.5)
        self.decoder_fc = nn.Linear(z_dim+c_dim, vocab_size)

        # discriminator
        self.conv1 = nn.Conv2d(1, 100, (3, self.emb_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, self.emb_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, self.emb_dim))
        self.disc_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(300, 2))

        self.discriminator = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.disc_fc])

        # bundle up model parameters
        self.encoder_params = chain(self.encoder.parameters(),
                                    self.q_mu.parameters(),
                                    self.q_logvar.parameters())

        self.decoder_params = chain(self.decoder.parameters(),
                                    self.decoder_fc.parameters())

        self.vae_params = chain(self.embedder.parameters(), self.encoder_params, self.decoder_params)

        # only keep model params that are differentiable
        self.vae_params = filter(lambda t: t.requires_grad, self.vae_params)

        self.discriminator_params = filter(lambda t: t.requires_grad, self.discriminator.parameters())


    def forwardEncoder(self, inputs):
        """
        Passes batch of sentences through embedding layer
        :param inputs:
        :return:
        """
        inputs = self.embedder(inputs)
        return self.forwardEncoderEmb(inputs)

    def forwardEncoderEmb(self, inputs):
        """
        Passes embeddings through encoder
        :param inputs:
        :return: mean and log-variance of input distribution
        """
        _, h = self.encoder(inputs, None)

        # pass latent space through mu and logvar layers
        h = h.view(-1, self.h_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar


    def sample_z(self, mu, logvar):
        """
        draws a z sample from the code distribution p_model(z)
        using the reparameterization trick z = mu + std * eps
        :param mu:
        :param logvar:
        :return:
        """
        eps = torch.rand(self.z_dim)
        return mu + torch.exp(logvar/2) * eps

    def sample_z_prior(self, size):
        z = torch.rand(size)
        return z

    def sample_c_prior(self, size):
        """
        samples controllable parameters
        for the decoder, where each parameter represents
        a tweet attribute we want to model (sentiment, tense).
        :return:
        """
        c = torch.from_numpy(np.random.multinomial(1, [0.5, 0.5], size).astype(np.float32))
        return c

    def forward_decoder(self, inputs, z, c):
        """
        passes embeddings, encoding, and controllable
        params through decoder to generate a new tweet
        :param inputs:
        :param z:
        :param c:
        :return:
        """
        decInputs = self.word_dropout(inputs)

        seqLen = decInputs.size(0)

        initH = torch.cat([z.unsqueeze(0), c.unsqueeze(0)], dim=2)
        inputsEmb = self.embedder(decInputs)
        inputsEmb = torch.cat([inputsEmb, initH.repeat(seqLen, 1, 1)], 2)

        outputs, _ = self.decoder(inputsEmb, initH)
        seqLen, size, _ = outputs.size()

        outputs = outputs.view(seqLen*size - 1)
        y = self.decoder_fc(outputs)
        y = y.view(seqLen, size, self.vocab_size)

        return y


