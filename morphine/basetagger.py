# -*- coding: utf-8 -*-
from __future__ import absolute_import
import abc
from operator import mul, add

import six
from six.moves import reduce
from pymorphy2.tokenizers import simple_word_tokenize

from morphine.feature_extractor import FeatureExtractor


def tokenize_if_needed(tokens):
    if not isinstance(tokens, (list, tuple)):
        return simple_word_tokenize(tokens)
    return tokens


class Disambiguator(object):
    """
    Combine several "partial" taggers (e.g. taggers for detecting
    word case, POS tag, number, gender) to assign probabilities
    to pymorphy2 parses.
    """
    def __init__(self, morph, partial_taggers, threshold=0):
        self.morph = morph
        self.partial_taggers = partial_taggers
        self.threshold = threshold

    def parse_sents(self, sents):
        return [self.parse(s) for s in sents]

    def parse(self, tokens):
        tokens, parsed_tokens = self._tokenize_and_parse(tokens)
        token_parse_probs = list(zip(*[
            tagger.predict_proba_single(tokens, parsed_tokens)
            for tagger in self.partial_taggers
        ]))

        res = []
        for parses, parse_probs in zip(parsed_tokens, token_parse_probs):
            probs = self._combine_marginals(parse_probs)
            scored_parses = [
                p._replace(score=prob) for p, prob in zip(parses, probs)
                if prob >= self.threshold
            ]
            scored_parses.sort(key=lambda p: p.score, reverse=True)
            res.append(scored_parses)

        return res

    def _tokenize_and_parse(self, sent_text):
        tokens = tokenize_if_needed(sent_text)
        parsed_tokens = [self.morph.parse(t) for t in tokens]
        return tokens, parsed_tokens

    def _combine_marginals(self, parse_marginals):
        # by default, multiply probabilities
        marginals = [reduce(mul, p) for p in zip(*parse_marginals)]

        # normalize to make them sum to 1
        k = sum(marginals)
        if k == 0:
            return [0.0] * len(marginals)
        else:
            return [m/k for m in marginals]


@six.add_metaclass(abc.ABCMeta)
class PartialTagger(object):
    """
    Base class for partial taggers.

    Each partial tagger should predict a value of a specific word
    attribute (POS tag, case, number, etc).
    """

    def __init__(self, feature_extractor, crf=None):
        """
        :param FeatureExtractor feature_extractor: Feature exractor object
        """
        self.fe = feature_extractor
        self.crf = crf

    @abc.abstractmethod
    def outval(self, tag):
        pass

    def predict_proba_single(self, tokens, parsed_tokens):
        if self.crf is None:
            raise ValueError("Tagger is not trained")

        xseq = self.fe.transform_single(
            self._prepared_tokens(tokens),
            parsed_tokens
        )
        marginals = self.crf.predict_marginals_single(xseq)
        return [
            [probs[self.outval(p.tag)] for p in parses]
            for parses, probs in zip(parsed_tokens, marginals)
        ]

    def _prepared_tokens(self, tokens):
        return tokenize_if_needed(tokens)


    # def predict(self, tokens, parsed_tokens):
    #     tokens = self._prepared_tokens(tokens)
    #     parsed_tokens, xseq = self.fe.transform_single(tokens, parsed_tokens)
    #     yseq = self.crf.tagger.tag(xseq)
    #     res = []
    #
    #     for i, (parses, value) in enumerate(zip(parsed_tokens, yseq)):
    #         p = max(parses, key=lambda p: self.outval(p.tag) == value)
    #         if value != 'NA':
    #             p = p._replace(score=self.crf.tagger.marginal(value, i))
    #         res.append(p)
    #
    #     return res



