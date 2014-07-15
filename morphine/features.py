# -*- coding: utf-8 -*-
from __future__ import absolute_import
import functools
from morphine.utils import func_takes_argument


def skips_empty_sents(func):
    @functools.wraps(func)
    def wrapper(tokens, parsed_tokens, feature_dicts):
        if not tokens:
            return
        return func(tokens, parsed_tokens, feature_dicts)
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
    # TODO: weighting: max / sum / one-zero / ...?
    def __init__(self, threshold=0.0, add_unambig=False):
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
                # TODO/FIXME: sum instead of max or in addition to max
                features[key] = max(p.score, features.get(key, 0))

                # TODO/FIXME: grammeme is unambiguous when its scores sums to 1
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
            grammemes = p.tag._grammemes_tuple
            for idx, grammeme in enumerate(grammemes):
                for grammeme2 in grammemes[idx+1:]:

                    # make grammeme order always the same
                    if grammeme < grammeme2:
                        key2 = "GrammemePair:%s,%s" % (grammeme, grammeme2)
                    else:
                        key2 = "GrammemePair:%s,%s" % (grammeme2, grammeme)

                    # TODO/FIXME: sum instead of max or in addition to max
                    features[key2] = max(p.score, features.get(key2, 0))

                    # TODO/FIXME: grammeme is unambiguous when its scores sums to 1
                    if self.add_unambig and p.score == 1:
                        features['unambig-'+key2] = 1

        return features


class Pattern(object):
    """
    Global feature that combines local features.
    """
    separator = '/'
    out_value = '?'
    missing_value = 0.0

    def __init__(self, *patterns, **kwargs):
        self.patterns = []
        names = []
        self.index_low = 0
        self.index_high = 0

        for pattern in patterns:
            if len(pattern) == 2:
                offset, feat = pattern
                if not isinstance(offset, int):
                    raise ValueError("Offset must be integer")

                name_template = '%s[%s]' if offset <= 0 else '%s[+%s]'
                func_name = feat.__name__ if callable(feat) else feat
                name = name_template % (func_name, offset)
            elif len(pattern) == 3:
                offset, feat, name = pattern
                if not isinstance(offset, int):
                    raise ValueError("Offset must be integer")
            else:
                raise ValueError("Patterns must be tuples with 2 or 3 elements")


            if callable(feat):
                if not func_takes_argument(feat, 'feature_dict'):
                    # fix "standard" token features
                    @functools.wraps(feat)
                    def _func(token, parses, feature_dict):
                        return feat(token, parses)
                    self.patterns.append((offset, _func, name))
                else:
                    self.patterns.append((offset, feat, name))

            else:
                # dictionary lookup
                def _lookup(token, parses, feature_dict, feat=feat):
                    return feature_dict.get(feat, self.missing_value)
                self.patterns.append((offset, _lookup, name))

            names.append(name)
            if self.index_low < -offset:
                self.index_low = -offset
            if self.index_high < offset:
                self.index_high = offset

        self.name = kwargs.pop('name', None)
        if self.name is None:
            self.name = self.separator.join(names)

        if 'index_low' in kwargs:
            self.index_low = kwargs['index_low']

        if 'index_high' in kwargs:
            self.index_high = kwargs['index_high']

    def __call__(self, tokens, parsed_tokens, feature_dicts):
        sliced = feature_dicts[self.index_low:]
        if self.index_high:
            sliced = sliced[:-self.index_high]
        for pos, featdict in enumerate(sliced, start=self.index_low):
            names = []
            values = []
            for offset, func, name in self.patterns:
                index = pos + offset
                if 0 <= index < len(feature_dicts):
                    value = func(tokens[index], parsed_tokens[index], feature_dicts[index])
                else:
                    value = self.out_value
                values.append(value)
                names.append(name)

            if len(self.patterns) == 1:
                featdict[self.name] = values[0]
            else:
                for value in values:
                    if isinstance(value, float) and value not in {0.0, 1.0}:
                        raise ValueError("Values must be boolean or string for Pattern to work")

                # FIXME: Python 2 unicode
                featdict[self.name] = self.separator.join(map(str, values))
