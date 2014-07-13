# -*- coding: utf-8 -*-
from __future__ import absolute_import
import functools


def skips_empty_sents(func):
    @functools.wraps(func)
    def wrapper(sent):
        if not sent:
            return
        return func(sent)
    return wrapper


def single_value(func):
    key = func.__name__
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return {key: func(*args, **kwargs)}
    return wrapper


@skips_empty_sents
def sentence_start(tokens, parsed_tokens, feature_dicts):
    feature_dicts[0]['sentence_start'] = 1.0


@skips_empty_sents
def sentence_end(tokens, parsed_tokens, feature_dicts):
    feature_dicts[-1]['sentence_end'] = 1.0


@single_value
def bias(token, parses):
    return 1.0


@single_value
def token_identity(token, parses):
    return token


@single_value
def token_lower(token, parses):
    return token.lower()


class _GrammemeFeatures(object):
    def __init__(self, threshold, add_unambig=False):
        self.threshold = threshold
        self.add_unambig = add_unambig

    def __call__(self, token, parses):
        parses = [p for p in parses if p.score >= self.threshold]
        return self.extract(parses)

    def extract(self, parses):
        raise NotImplementedError()


class Grammeme(_GrammemeFeatures):
    """
    This feature adds all seen grammemes with their weights to the result.
    If there are several weights possible, the maximum is used.
    """
    def extract(self, parses):
        features = {}
        for p in parses:
            for grammeme in p.tag.grammemes:
                key = "Grammeme:" + grammeme
                features[key] = max(p.score, features[key])

                if self.add_unambig and p.score == 1:
                    features['unambig-'+key] = 1
        return features


class GrammemePair(_GrammemeFeatures):
    """
    This feature adds all seen grammeme pairs with their weights to the result.
    If there are several weights possible, the maximum is used.
    """
    def extract(self, parses):
        features = {}
        for p in parses:
            for grammeme in p.tag.grammemes:
                seen = set()
                for grammeme2 in p.tag.grammemes:
                    if grammeme2 == grammeme:
                        continue

                    if grammeme > grammeme2:
                        key2 = "GrammemePair:%s,%s" % (grammeme, grammeme2)
                    else:
                        key2 = "GrammemePair:%s,%s" % (grammeme2, grammeme)

                    if key2 in seen:
                        continue
                    seen.add(key2)
                    features[key2] = max(p.score, features[key2])

                    if self.add_unambig and p.score == 1:
                        features['unambig-'+key2] = 1
