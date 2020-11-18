import os
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


# load vocab
with open('./data/vocab.txt', 'r') as file:
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

