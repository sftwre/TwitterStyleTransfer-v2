import torch
import numpy as np
import pandas as pd
from typing import List
from collections import Counter
from utils import *

class TwitterDataset():

    def __init__(self, batch_size=32, gpu=False):

        self.gpu = gpu
        self.batch_size = batch_size

        vocabPath = '../data/vocab.txt'
        trainPath = '../data/tweets.train.txt'
        trainLabelsPath = '../data/tweets.train.labels'

        testPath = '../data/tweets.test.txt'
        testLabelsPath = '../data/tweets.test.labels'

        # load vocab
        self.vocab = self._loadVocab(vocabPath)

        # load training data
        self.X_train, self.y_train = self._loadData(trainPath, trainLabelsPath)

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
        tweet_indexer.add_and_get_index(UNK_SYMBOL)
        tweet_indexer.add_and_get_index(PAD_SYMBOL)

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

    def _batchToIdxs(self, batch:List[str], labels=False) -> torch.Tensor:
        """
        converts batch of tweets into corresponding indices
        :param batch:
        :return:
        """
        idxs = list()

        indexer = self.tweet_indexer if not labels else self.account_indexer

        for ex in batch:
            idxs.append(list(map(lambda x: indexer.index_of(x), ex)))

        return torch.tensor(idxs)


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
            encodedText = self._batchToIdxs(text[start:end])
            encodedLabels = self._batchToIdxs(labels[start:end], labels=True)

            if self.gpu:
                encodedText = encodedText.cuda()
                encodedLabels = encodedLabels.cuda()

            yield encodedText, encodedLabels

    def resetTrainBatches(self):
        self.trainIterator = self._dataIterator(self.X_train, self.y_train)

    def idxs2sentence(self, idxs):
        return ' '.join([self.tweet_indexer.get_object(i) for i in idxs])

    def idx2label(self, idx):
        return self.account_indexer.get_object(idx)







