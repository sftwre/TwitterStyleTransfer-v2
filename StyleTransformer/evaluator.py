"""
Modified code from https://github.com/fastnlp/style-transformer
"""

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

import torch
import fasttext
import pkg_resources
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import math


class Evaluator(object):

    def __init__(self):
        resource_package = __name__

        twitter_acc_path = 'acc_twitter.bin'
        twitter_ppl_path = 'ppl_twitter.binary'
        twitter_ref0_path = 'twitter.refs.0'
        twitter_ref1_path = 'twitter.refs.1'

        twitter_acc_file = pkg_resources.resource_stream(resource_package, twitter_acc_path)
        twitter_ppl_file = pkg_resources.resource_stream(resource_package, twitter_ppl_path)
        twitter_ref0_file = pkg_resources.resource_stream(resource_package, twitter_ref0_path)
        twitter_ref1_file = pkg_resources.resource_stream(resource_package, twitter_ref1_path)

        self.twitter_ref = []
        with open(twitter_ref0_file.name, 'r') as fin:
            self.twitter_ref.append(fin.readlines())
        with open(twitter_ref1_file.name, 'r') as fin:
            self.twitter_ref.append(fin.readlines())
        self.classifier_twitter = fasttext.load_model(twitter_acc_file.name)

        self.twitter_ppl_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

        self.twitter_ppl_model.eval()

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def twitter_style_check(self, text_transfered, style_origin):
        text_transfered = ' '.join(word_tokenize(text_transfered.lower().strip()))
        if text_transfered == '':
            return False
        label = self.classifier_twitter.predict([text_transfered])
        style_transfered = label[0][0] == '__label__positive'
        return (style_transfered != style_origin)

    def ppl_score(self, sentence):

        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = self.twitter_ppl_model(tensor_input, lm_labels=tensor_input)
        return math.exp(loss)

    def twitter_acc_b(self, texts, styles_origin):
        assert len(texts) == len(styles_origin), 'Size of inputs does not match!'
        count = 0
        for text, style in zip(texts, styles_origin):
            if self.twitter_style_check(text, style):
                count += 1
        return count / len(texts)

    def twitter_acc_0(self, texts):
        styles_origin = [0] * len(texts)
        return self.twitter_acc_b(texts, styles_origin)

    def twitter_acc_1(self, texts):
        styles_origin = [1] * len(texts)
        return self.twitter_acc_b(texts, styles_origin)

    def nltk_bleu(self, texts_origin, text_transfered):
        texts_origin = [word_tokenize(text_origin.lower().strip()) for text_origin in texts_origin]
        text_transfered = word_tokenize(text_transfered.lower().strip())
        return sentence_bleu(texts_origin, text_transfered) * 100

    def self_bleu_b(self, texts_origin, texts_transfered):
        assert len(texts_origin) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_origin)
        for x, y in zip(texts_origin, texts_transfered):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def twitter_ref_bleu_0(self, texts_neg2pos):
        assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 500
        for x, y in zip(self.twitter_ref[0], texts_neg2pos):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def twitter_ref_bleu_1(self, texts_pos2neg):
        assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 500
        for x, y in zip(self.twitter_ref[1], texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def twitter_ref_bleu(self, texts_neg2pos, texts_pos2neg):
        assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 1000
        for x, y in zip(self.twitter_ref[0] + self.twitter_ref[1], texts_neg2pos + texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def twitter_ppl(self, texts_transfered):
        texts_transfered = [' '.join(word_tokenize(itm.lower().strip())) for itm in texts_transfered]
        sum = 0
        words = []
        length = 0
        for i, line in enumerate(texts_transfered):
            words += [word for word in line.split()]
            length += len(line.split())
            score = self.ppl_score(line)
            sum += score
        return math.pow(10, -sum / length)


