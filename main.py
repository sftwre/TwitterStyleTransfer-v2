import os
import nltk
import string
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.layers import Embedding
import tensorflow.keras.preprocessing as pre
nltk.download('stopwords')

# max chars in tweet
TWEET_LEN = 280
trainData = './data/dril.train.txt'

# load vocab
with open('./data/vocab.txt', 'r', encoding='UTF-8') as file:
    vocab = file.read().split()
    vocab = set(vocab)

# stopwords set
sw = set(stopwords.words('english'))

def cleanTweet(tweet):
    # split into sentences
    tokens = tweet.split()

    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = list(map(lambda x: x.translate(table), tokens))

    # filter out non-alphabetic, stopwords, short tokens, and tokens not in vocab
    tokens = list(filter(lambda x: x.isalpha() and x not in sw \
                                   and len(x) > 1 and x in vocab, tokens))

    # create new sentences
    tokens = ' '.join(tokens)
    return tokens

# load training data
with open(trainData, 'r', encoding='UTF-8') as file:
    X_train = file.read().split('\n')

# clean training data
X_train = list(map(cleanTweet, X_train))

# map words to integers
tokenizer = pre.text.Tokenizer()
tokenizer.fit_on_texts(X_train)

# encode data
encodedTweets = tokenizer.texts_to_sequences(X_train)

X_train = pre.sequence.pad_sequences(encodedTweets, maxlen=TWEET_LEN, padding='post')

# add dimension for sequence length, 1
X_train = np.expand_dims(X_train, 1).astype(np.float32)

# create embedding layer
embedding = Embedding(input_dim=len(vocab), output_dim=100, input_length=TWEET_LEN)