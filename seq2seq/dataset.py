import torch
import numpy as np
import pandas as pd
from typing import List
from collections import Counter
from seq2seq.utils import *

class TwitterDataset():

    def __init__(self, batch_size, vocab_path, train_path, labels_path):

        self.batch_size = batch_size
        self.max_input_len = -1

        # load vocab
        self.vocab = self._loadVocab(vocab_path)

        # load training data
        self.X_train, self.y_train = self._loadData(train_path, labels_path)

        self.tweet_indexer = self._buildTweetVocab()
        self.account_indexer = self._buildAccountVocab(self.y_train)

        # size of vocabulary
        self.vocab_size = len(self.vocab)

        # number of accounts dataset
        self.n_accounts = len(self.account_indexer)

        # define training iterator
        self.trainIterator = None

    def _loadData(self, dataFile, labelsFile):
        """
        Loads twitter data and labels and applies filters.
        """
        # first element is twitter data, second are labels
        data = []
        for path in [dataFile, labelsFile]:
            with open(path, 'r', encoding='UTF-8') as file:
                text = file.read().split('\n')
                data.append(text)

        # remove tokens not in the vocab
        data[0] = list(map(self._cleanTweet, data[0]))

        df = pd.DataFrame(data={'tweet': data[0], 'account': data[1]})

        # remove empty tweets
        df = df[df.tweet != '']

        return df.tweet.values, df.account.values

    def _loadVocab(self, path):
        """
        Loads vocabulary file
        :param path:
        :return:
        """
        with open(path, 'r', encoding='UTF-8') as file:
            vocab = file.read().split()
            vocab = set(vocab)

        return vocab

    def _cleanTweet(self, tweet):
        # split into sentences
        tokens = tweet.strip().lower().split()

        # filter out words not in vocabulary
        tokens = list(filter(lambda x: x.isalpha() and x in self.vocab, tokens))

        # create new sentences
        return ' '.join(tokens)

    def _buildTweetVocab(self):
        """
        adds words in vocab to indexer and returns it
        Assumes the vocab has been loaded into memory
        :return:
        """

        tweet_indexer = Indexer()

        # add special symbols
        tweet_indexer.add_and_get_index(PAD_SYMBOL)
        tweet_indexer.add_and_get_index(UNK_SYMBOL)
        tweet_indexer.add_and_get_index(SOS_SYMBOL)
        tweet_indexer.add_and_get_index(EOS_SYMBOL)

        # add each word to indexer
        for word in self.vocab:
            tweet_indexer.add_and_get_index(word)

        return tweet_indexer

    def _buildAccountVocab(self, acct_handles:np.ndarray):
        """
        adds account handles to indexer and returns it
        :param acct_handles:
        :return:
        """
        account_indexer = Indexer()

        account_indexer.add_and_get_index(UNK_SYMBOL)

        for acct in acct_handles:
            account_indexer.add_and_get_index(acct)

        return account_indexer

    def _batchToIdxs(self, batch:List[List[str]], max_len) -> torch.Tensor:
        """
        Converts batch of tweets  into corresponding indices in the input indexer
        and pads the inputs to a fixed length.
        :param batch: batch of inputs or account handels
        :param max_len: max length of tweet in training data, used to create fixed length vector.
        :return: tensor of padded indices
        """
        inputs = list()

        for ex in batch:
            inputs.append(list(map(lambda x: self.tweet_indexer.index_of(x), ex)))

        # pad inputs to max len
        padded_inputs = np.array([[ex[i] if i < len(ex) else self.tweet_indexer.index_of(PAD_SYMBOL) for i in range(max_len)] for ex in inputs])
        inputs = torch.tensor(padded_inputs)

        return inputs

    def _labelsToIdxs(self, labels) -> torch.Tensor:
        """
        Converts account handles to corresponding indices
        """
        inputs = list()
        inputs.append(list(map(lambda x: self.account_indexer.index_of(x), labels)))
        inputs = torch.tensor(inputs).reshape((-1,1))
        return inputs

    def _dataIterator(self, text, labels):
        """
        Generator that yields batch_size items from a dataset
        sorted by increasing length.
        :param data:
        :param batch_size:
        :return:
        """
        try:
            assert len(text) == len(labels)
        except AssertionError:
            print('Length of text data and labels must match',
                  f'text length: {len(text)}, labels length: {len(labels)}', sep='\n')
            exit(-1)

        # sort tweets by length
        text_labels = list(zip(text, labels))
        text_labels = sorted(text_labels, key=lambda x: len(x[0]), reverse=False)
        tuples = zip(*text_labels)

        text, labels = [list(t) for t in tuples]

        # generate batches
        for start in range(0, len(text), self.batch_size):
            end = start + self.batch_size

            batch_tweets = [ex.split() for ex in text[start:end]]
            batch_labels = labels[start:end]

            padded_inputs = self._batchToIdxs(batch_tweets, self.max_input_len)
            idx_labels = self._labelsToIdxs(batch_labels)

            # shuffle batch
            idxs = np.arange(padded_inputs.shape[0])
            np.random.shuffle(idxs)
            padded_inputs = padded_inputs[idxs, :]
            idx_labels = idx_labels[idxs, :]


            yield padded_inputs, idx_labels

    def resetTrainBatches(self):

        # set max sequence length
        if self.max_input_len < 0:
            self.max_input_len = max(list(map(lambda x: len(x.split()), self.X_train)))

        self.trainIterator = self._dataIterator(self.X_train, self.y_train)

    def idxs2sentence(self, idxs):
        return ' '.join([self.tweet_indexer.get_object(i) for i in idxs])

    def idx2label(self, idx):
        return self.account_indexer.get_object(idx)







