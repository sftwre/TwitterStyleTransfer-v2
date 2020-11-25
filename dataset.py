import nltk
import string
import numpy as np
from torchtext import data
from nltk.corpus import stopwords
nltk.download('stopwords')

class TwitterDataset():

    def __init__(self, emb_dim=50, batch_size=32, tweet_len=280):
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=tweet_len)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        vocabPath = './data/vocab.txt'
        trainPath = './data/tweets.train.txt'
        trainLabelsPath = './data/tweets.train.labels'

        testPath = './data/tweets.test.txt'
        testLabelsPath = './data/tweets.test.labels'

        # stopwords to remove from tweets
        self.sw = set(stopwords.words('english'))

        # load vocab
        self.vocab = self._loadVocab(vocabPath)

        # load training data
        X_train, y_train = self._loadData(trainPath, trainLabelsPath)

        self.X_train = X_train
        self.y_train = y_train

        self.TEXT.build_vocab(self.vocab)
        self.LABEL.build_vocab(y_train)

        # number of characters in vocab
        self.n_vocab = len(self.TEXT.vocab.itos)

        # define training iterator
        self.train_iter = self._dataIterator(self.X_train, self.y_train)

    def _loadData(self, dataFile, labelsFile):
        """
        Loads twitter data and labels and apply filters.
        """
        # first element is twitter data, second are labels
        data = []
        for path in [dataFile, labelsFile]:
            with open(path, 'r', encoding='UTF-8') as file:
                text = file.read().split('\n')
                data.append(text)

        # filter tweets and remove empty tweets
        data[0] = np.array(list(map(self._cleanTweet, data[0])))
        data[0] = list(filter(lambda x: len(x) > 0, data[0]))

        return np.array(data)

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
        tokens = tweet.split()

        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = list(map(lambda x: x.translate(table), tokens))

        # filter out non-alphabetic, stopwords, short tokens, and tokens not in vocab
        tokens = list(filter(lambda x: x.isalpha() and x not in sw \
                                       and len(x) > 1 and x in self.vocab, tokens))

        # create new sentences
        tokens = ' '.join(tokens)
        return tokens

    def _dataIterator(self, text, labels):
        """
        Generator that yields batch_size items from a dataset
        :param data:
        :param batch_size:
        :return:
        """
        try:
            assert len(text) == len(labels)
        except AssertionError:
            print(f'Length of text data and labels must \
            match \n text length: {len(text)}, labels length: {len(labels)}')
            exit(-1)

        for start in range(0, len(text), self.batch_size):
            end = start + self.batch_size
            yield text[start:end], labels[start:end]

    def nextBatch(self):
        tweets, labels = next(self.train_iter)
        return tweets, labels

    def getVocabVectors(self):
        return self.TEXT.vocab.vectors

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]







