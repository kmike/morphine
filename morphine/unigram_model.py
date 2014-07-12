# -*- coding: utf-8 -*-
from __future__ import absolute_import
from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize


class Tagger(object):
    def __init__(self, morph=None):
        if morph is None:
            morph = MorphAnalyzer()
        self.morph = morph

    def predict(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            tokens = simple_word_tokenize(tokens)
        return [self.morph.parse(token)[0] for token in tokens]
