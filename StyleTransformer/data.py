"""
Modified code from https://github.com/fastnlp/style-transformer
"""
import os
import time
import numpy as np
import torchtext
from torchtext import data
# from utils import tensor2text


class DatasetIterator(object):
    def __init__(self, elon_iter, dalai_iter):
        self.elon_iter = elon_iter
        self.dalai_iter = dalai_iter

    def __iter__(self):

        for batch_elon, batch_dalai in zip(iter(self.elon_iter), iter(self.dalai_iter)):

            if batch_elon.text.size(0) == batch_dalai.text.size(0):
                yield batch_elon.text, batch_dalai.text


def load_dataset(config, train_elon='elonmusk.txt', train_dalai='DalaiLama.txt',
                 test_biden='JoeBiden.txt', test_dril='dril.txt'):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token='<eos>')

    dataset_fn = lambda name: data.TabularDataset(
        path=os.path.join(root, name),
        format='tsv',
        fields=[('text', TEXT)]
    )

    train_elon_set, train_dalai_set = map(dataset_fn, [train_elon, train_dalai])
    test_biden_set, test_dril_set = map(dataset_fn, [test_biden, test_dril])

    TEXT.build_vocab(train_elon_set, train_dalai_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()

        vectors = torchtext.vocab.GloVe('6B', dim=config.embed_size, cache=config.pretrained_embed_path)
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print('vectors', TEXT.vocab.vectors.size())

        print('load embedding took {:.2f} s.'.format(time.time() - start))

    vocab = TEXT.vocab

    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device
    )

    train_elon_iter, train_dalai_iter = map(lambda x: dataiter_fn(x, True), [train_elon_set, train_dalai_set])
    test_biden_iter, test_dril_iter = map(lambda x: dataiter_fn(x, False), [test_biden_set, test_dril_set])

    train_iters = DatasetIterator(train_elon_iter, train_dalai_iter)
    test_iters = DatasetIterator(test_biden_iter, test_dril_iter)

    return train_iters, test_iters, vocab

# if __name__ == '__main__':
    # train_iter, _, _, vocab = load_dataset('../data/yelp/')
    # print(len(vocab))
    # for batch in train_iter:
    #     text = tensor2text(vocab, batch.text)
    #     print('\n'.join(text))
    #     print(batch.label)
    #     break
