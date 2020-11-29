import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class VAE(nn.Module):

    def __init__(self, vocab_size, h_dim, z_dim, c_dim, n_accounts, gpu=False):
        super(VAE, self).__init__()


        self.gpu = gpu
        self.emb_dim = h_dim
        self.z_dim = z_dim
        self.vocab_size = vocab_size
        self.n_accounts = n_accounts
        self.p_word_dropout = 0.5
        self.unk_idx = 0
        self.pad_idx = 1
        self.start_idx = 2
        self.eos_idx = 3
        self.max_tweet_len = 280
        self.gpu = gpu

        # embedding layer
        self.embedder = nn.Embedding(vocab_size, h_dim, self.pad_idx)

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
        self.decoder_fc1 = nn.Linear(z_dim + c_dim, vocab_size)
        self.decoder_fc2 = nn.Sequential(nn.Linear(vocab_size, n_accounts, bias=False), nn.Softmax(dim=2))

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
                                    self.decoder_fc1.parameters())

        self.vae_params = chain(self.embedder.parameters(), self.encoder_params, self.decoder_params)

        # only keep model params that are differentiable
        self.vae_params = filter(lambda t: t.requires_grad, self.vae_params)

        self.discriminator_params = filter(lambda t: t.requires_grad, self.discriminator.parameters())

        if self.gpu:
            self.cuda('cuda:0')

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
        _, (h, c) = self.encoder(inputs, None)

        # pass latent space through mu and logvar layers
        h = h.view(-1, self.emb_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar, c


    def sample_z(self, mu, logvar):
        """
        draws a z sample from the code distribution p_model(z)
        using the reparameterization trick z = mu + std * eps
        :param mu:
        :param logvar:
        :return:
        """
        eps = torch.rand(self.z_dim)
        eps = eps.cuda() if self.gpu else eps
        return mu + torch.exp(logvar/2) * eps

    def sample_z_prior(self, size):
        z = torch.rand(size, self.z_dim)
        z = z.cuda() if self.gpu else z
        return z

    def sample_c_prior(self, size):
        """
        samples controllable parameters
        for the decoder, where each parameter represents
        a tweet attribute we want to model (sentiment, tense).
        :return:
        """
        c = torch.from_numpy(np.random.multinomial(1, [0.5, 0.5], size).astype(np.float32))
        c = c.cuda() if self.gpu else c
        return c

    def forwardDecoder(self, inputs, z, c, initC):
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
        initC = torch.cat([initC, c.unsqueeze(0)], dim=2)
        inputsEmb = self.embedder(decInputs)
        inputsEmb = torch.cat([inputsEmb, initH.repeat(seqLen, 1, 1)], 2)

        outputs, _ = self.decoder(inputsEmb, (initH, initC))
        seqLen, bsize, _ = outputs.size()

        outputs = outputs.view(seqLen*bsize, -1)

        # y1 is prob. distribution over vocabulary
        y1 = self.decoder_fc1(outputs)
        y1 = y1.view(seqLen, bsize, self.vocab_size)

        # y2 is prob. distribution over accounts
        y2 = self.decoder_fc2(y1)
        return y1, y2

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

    def forward(self, inputs, labels, use_c_prior=True):
        """
        tweet: sequence of word indices.
        use_c_prior: whether to sample c from prior or from discriminator.

        Returns: reconstruction loss of VAE and KL-div loss of VAE.
        """
        self.train()

        size = inputs.size(1)

        pad_words = torch.LongTensor([1]).repeat(1, size)
        pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = inputs
        dec_inputs = inputs
        dec_targets = torch.cat([inputs[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar, cell_state = self.forwardEncoder(enc_inputs)
        z = self.sample_z(mu, logvar)

        if use_c_prior:
            c = self.sample_c_prior(size)
        else:
            c = self.forwardDiscriminator(inputs.transpose(0, 1))

        # Decoder: sentence -> y
        y1, y2 = self.forwardDecoder(dec_inputs, z, c, cell_state)

        recon_loss = F.cross_entropy(y1.view(-1, self.vocab_size), dec_targets.view(-1), size_average=True)
        thandle_loss = F.cross_entropy(y2.view(-1, self.n_accounts), labels)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))

        return recon_loss, thandle_loss, kl_loss

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

    def sample_sentence(self, z, c, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        self.eval()

        word = torch.LongTensor([self.start_idx])
        word = word.cuda() if self.gpu else word

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        h = torch.cat([z, c], dim=2)
        c_0 = torch.zeros(h.shape)

        if self.gpu:
            c_0 = c_0.cuda()

        state = (h, c_0)

        outputs = []

        if raw:
            outputs.append(self.start_idx)

        for i in range(self.max_tweet_len):
            emb = self.embedder(word).view(1, 1, -1)
            emb = torch.cat([emb, z, c], 2)

            output, h = self.decoder(emb, state)
            y = self.decoder_fc1(output).view(-1)
            y = F.softmax(y/temp, dim=0)

            idx = torch.multinomial(y, 1)

            word = torch.LongTensor([int(idx)])
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.eos_idx:
                break

            outputs.append(idx)

        # Back to default state: train
        self.train()

        if raw:
            outputs = torch.LongTensor(outputs).unsqueeze(0)
            return outputs.cuda() if self.gpu else outputs
        else:
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
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """
        self.eval()

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = word # '<start>'
        emb = self.embedder(word).view(1, 1, -1)
        emb = torch.cat([emb, z, c], 2)

        h = torch.cat([z, c], dim=2)

        outputs = [self.embedder(word).view(1, -1)]

        for i in range(self.max_tweet_len):
            output, h = self.decoder(emb, h)
            o = self.decoder_fc1(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.word_emb.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with z and c for the next input
            emb = torch.cat([emb, z, c], 2)

        # 1 x 16 x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        # Back to default state: train
        self.train()

        return outputs.cuda() if self.gpu else outputs

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size())).astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.unk_idx

        return data


