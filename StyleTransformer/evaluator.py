"""
Modified code from https://github.com/fastnlp/style-transformer
"""

from nltk.tokenize import word_tokenize
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import math


class Evaluator(object):

    def __init__(self):

        self.twitter_ppl_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.twitter_ppl_model.eval()

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def ppl_score(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = self.twitter_ppl_model(tensor_input, lm_labels=tensor_input)
        return math.exp(loss.item())


    def twitter_ppl(self, texts_transfered):
        texts_transfered = [' '.join(word_tokenize(itm.lower().strip())) for itm in texts_transfered]
        sum = 0
        words = []
        length = 0
        for i, line in enumerate(texts_transfered):

            # skip empty input
            if len(line) == 0:
                continue

            words += [word for word in line.split()]
            length += len(line.split())
            score = self.ppl_score(line)
            sum += score
        return sum / length


