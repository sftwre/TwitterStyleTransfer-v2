import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from utils import *

class VAE(nn.Module):

    def __init__(self, indexer, h_dim=64, z_dim=64, c_dim=4):
        super(VAE, self).__init__()

        self.emb_dim = h_dim
        self.z_dim = z_dim
        self.output_sz = len(indexer)
        self.c_dim = c_dim
        self.indexer = indexer
        self.p_word_dropout = 0.5
        self.unk_idx = self.indexer.index_of(UNK_SYMBOL)
        self.pad_idx = self.indexer.index_of(PAD_SYMBOL)
        self.start_idx = self.indexer.index_of(SOS_SYMBOL)
        self.eos_idx = self.indexer.index_of(EOS_SYMBOL)
        self.max_tweet_len = 15

        # TODO load ElMo embeddings
        # embedding layer
        self.embedder = nn.Embedding(self.output_sz, self.emb_dim)

        """
        encoder is uni-directional LSTM with fully connected 
        layer on top of last hidden unit to model the mean
        and log variance of latent space.
        """
        self.encoder = nn.LSTM(self.emb_dim, h_dim, num_layers=1, batch_first=True)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)

        """
        decoder is LSTM with embeddings, z, and c as inputs
        """
        self.decoder = nn.LSTM(self.emb_dim+z_dim+c_dim, z_dim+c_dim, num_layers=1, batch_first=True)
        self.decoder_fc = nn.Linear(z_dim + c_dim, self.output_sz)

        # discriminator
        self.conv1 = nn.Conv2d(1, 100, (3, self.emb_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, self.emb_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, self.emb_dim))
        self.disc_fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(300, c_dim))

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


    def forwardEncoder(self, inputs, input_lens):
        """
        Passes batch of sentences through embedding layer
        and then through the encoder.
        :param inputs:
        :return:
        """
        inputs = self.embedder(inputs)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(inputs, input_lens, batch_first=True, enforce_sorted=False)
        _, (h, c) = self.encoder(packed_embeddings)

        # pass latent space through mu and logvar layers
        h = h.reshape(-1, self.emb_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar, c


    def sample_z(self, mu, logvar):
        """
        draws a z sample from the code distribution p_model(z)
        using the re-parameterization trick z = mu + std * eps
        :param mu:
        :param logvar:
        :return:
        """
        eps = torch.rand(self.z_dim)
        # eps = eps.cuda() if self.gpu else eps
        return mu + torch.exp(logvar/2) * eps

    def sample_z_prior(self, size):
        z = torch.rand(size, self.z_dim)
        # z = z.cuda() if self.gpu else z
        return z

    def sample_c_prior(self, size):
        """
        samples controllable parameters
        for the decoder, where each parameter represents
        a tweet attribute we want to model (twitter handle).
        :return:
        """
        n = self.c_dim
        c = torch.from_numpy(np.random.multinomial(1, [1./n]*n, size).astype(np.float32))
        # c = c.cuda() if self.gpu else c
        return c

    def forwardDecoder(self, inputs, z, c, init_c):
        """
        passes embeddings, encoding, and controllable
        params through decoder to generate a new tweet
        :param inputs:
        :param z:
        :param c:
        :return:
        """

        # TODO test model with word dropout
        # decInputs = self.word_dropout(inputs)

        # TODO create more sophisticated Encoder/Decoder

        # re-construction loss
        recon_loss = 0.0
        batch_sz = inputs.shape[0]
        target_len = inputs.shape[1]

        # latent and parameter code are initial hidden state of decoder
        init_h = torch.cat([z.unsqueeze(0), c.unsqueeze(0)], dim=2)
        init_c = torch.cat([init_c, c.unsqueeze(0)], dim=2)

        # initial hidden state
        h_n = (init_h, init_c)

        # initial input token
        y_step = torch.tensor([[self.indexer.index_of(SOS_SYMBOL)] * batch_sz]).reshape(batch_sz, 1)

        # predict next token using teacher forcing
        for t in range(target_len):

            word_embeddings = self.embedder(y_step)

            # concat new latent and parameter code
            z_c = h_n[0].permute((1, 0, 2))
            dec_inputs = torch.cat([word_embeddings, z_c], dim=2)

            outputs, h_n = self.decoder(dec_inputs, h_n)

            logits = self.decoder_fc(outputs).reshape(-1, self.output_sz)
            log_probs = F.log_softmax(logits, dim=1)

            target = inputs[:, t]
            recon_loss += F.nll_loss(log_probs, target, ignore_index=self.pad_idx)

            # teacher forcing, next input is current target
            y_step = target.reshape(batch_sz, 1)

            # target_len, bsize, _ = outputs.size()
            #
            # outputs = outputs.reshape(target_len*bsize, -1)

        # logits for generated words
        # y = self.decoder_fc(outputs)
        # y = y.view(target_len, bsize, self.output_sz)

        # compute mean loss across time steps
        recon_loss /= target_len

        return recon_loss

    def forwardDiscriminator(self, inputs):
        """
        Inputs is batch of sentences size x seqlen
        :param inputs:
        :return:
        """
        inputs = self.embedder(inputs)
        return self.forwardDiscEmbed(inputs)

    def forwardDiscEmbed(self, inputs):
        """
        Inputs is embeddings of shape size x seqLen x emb_dim
        :param inputs:
        :return: classified attributes
        """
        inputs = inputs.unsqueeze(1)

        x1 = F.relu(self.conv1(inputs)).squeeze()
        x2 = F.relu(self.conv2(inputs)).squeeze()
        x3 = F.relu(self.conv3(inputs)).squeeze()

        x1 = F.max_pool1d(x1, x1.size(2)).squeeze()
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze()
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()

        x = torch.cat([x1, x2, x3], dim=1)

        c_hat = self.disc_fc(x)

        return c_hat

    def forward(self, inputs, input_lens, use_c_prior=True):
        """
        inputs: batch of padded inputs
        use_c_prior: whether to sample c from prior or from discriminator.

        Returns: reconstruction loss of VAE and KL-div loss of VAE.
        """
        self.train()

        size = inputs.shape[0]

        # pad_words = torch.LongTensor([1]).repeat(1, size)
        # pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = inputs
        dec_inputs = inputs
        # dec_targets = torch.cat([inputs[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar, cell_state = self.forwardEncoder(enc_inputs, input_lens)
        z = self.sample_z(mu, logvar)

        if use_c_prior:
            c = self.sample_c_prior(size)
        else:
            c = self.forwardDiscriminator(inputs.transpose(0, 1))

        recon_loss = self.forwardDecoder(dec_inputs, z, c, cell_state)

        # recon_loss = F.cross_entropy(y1.view(-1, self.output_sz), dec_targets.view(-1), size_average=True)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))

        return recon_loss, kl_loss

    def generate_sentences(self, batch_size):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        samples = []
        cs = []

        for _ in range(batch_size):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)
            samples.append(self.sample_sentence(z, c, raw=True))
            cs.append(c.long())

        X_gen = torch.cat(samples, dim=0)
        c_gen = torch.cat(cs, dim=0)

        return X_gen, c_gen

    def sample_sentence(self, z, c, raw=False, beam=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        self.eval()

        word = torch.LongTensor([self.start_idx])

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        h_0 = torch.cat([z, c], dim=2)
        c_0 = torch.zeros(h_0.shape)

        init_h = (h_0, c_0)

        outputs = []

        with torch.no_grad():

            if raw:
                outputs.append(self.start_idx)

            for i in range(self.max_tweet_len):
                emb = self.embedder(word).view(1, 1, -1)
                emb = torch.cat([emb, z, c], 2)

                output, h_0 = self.decoder(emb, init_h)
                logits = self.decoder_fc(output).view(-1)

                # TODO re-incorperate temperature annealing
                # y = F.softmax(y/temp, dim=0)
                log_probs = F.log_softmax(logits)
                y_hat = torch.argmax(log_probs)

                # idx = torch.multinomial(y_hat, 1)

                # word = torch.LongTensor([int(idx)])
                # word = word.cuda() if self.gpu else word

                # idx = int(idx)

                if not raw and y_hat == self.eos_idx:
                    break

                outputs.append(y_hat.item())

        # Back to default state: train
        self.train()

        if raw:
            outputs = torch.LongTensor(outputs).unsqueeze(0)

        return outputs

    def generate_soft_embed(self, mbsize, temp=1):
        """
        Generate soft embeddings of (mbsize x emb_dim) along with target z
        and c for each row (mbsize x {z_dim, c_dim})
        """
        samples = []
        targets_c = []
        targets_z = []

        for _ in range(mbsize):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)

            samples.append(self.sample_soft_embed(z, c, temp=1))
            targets_z.append(z)
            targets_c.append(c)

        X_gen = torch.cat(samples, dim=0)
        targets_z = torch.cat(targets_z, dim=0)
        _, targets_c = torch.cat(targets_c, dim=0).max(dim=1)

        return X_gen, targets_z, targets_c

    def sample_soft_embed(self, z, c, temp=1):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word embeddings
        according to p(x|z,c).

        Used to the generator during the sleep phase
        """
        # self.eval()

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        word = torch.LongTensor([self.start_idx])
        # word = word.cuda() if self.gpu else word
        # word = word # '<start>'
        emb = self.embedder(word).view(1, 1, -1)
        emb = torch.cat([emb, z, c], 2)

        h = torch.cat([z, c], dim=2)
        zeros = torch.zeros((1, 1, self.z_dim))

        c_0 = torch.cat([zeros, c], dim=2)

        state = (h, c_0)

        outputs = [self.embedder(word).view(1, -1)]

        for i in range(self.max_tweet_len):
            output, h = self.decoder(emb, state)
            o = self.decoder_fc(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.embedder.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with z and c for the next input
            emb = torch.cat([emb, z, c], 2)

        # 1 x 16 x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        # Back to default state: train
        # self.train()

        return outputs

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size())).astype('uint8')
        )

        # Set to <unk>
        data[mask] = self.unk_idx

        return data


