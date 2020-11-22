import os
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
import tensorflow.keras.preprocessing as pre

# max chars in tweet
TWEET_LEN = 280
trainData = './data/dril.train.txt'

# load vocab
with open('./data/vocab.txt', 'r', encoding='UTF-8') as file:
    vocab = file.read().split()
    vocab = set(vocab)

def cleanTweets(doc, vocab):
    # split into words
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = list(map(lambda x: x.translate(table), tokens))

    # filter out tokens not in vocab
    tokens = list(filter(lambda x: x in vocab, tokens))
    tokens = ' '.join(tokens)
    return tokens

# load training data
with open(trainData, 'r', encoding='UTF-8') as file:
    X_train = file.read()

# clean training data
X_train = cleanTweets(X_train, vocab)

# map words to integers
tokenizer = pre.text.Tokenizer()
tokenizer.fit_on_texts(X_train)

# encode data
encodedTweets = tokenizer.texts_to_sequences(X_train)

X_train = pre.sequence.pad_sequences(encodedTweets, maxlen=TWEET_LEN, padding='post')
vocabSize = len(tokenizer.word_index) + 1

print(X_train)

# create embedding layer
embedding = Embedding(input_dim=vocabSize, output_dim=100, input_length=TWEET_LEN)