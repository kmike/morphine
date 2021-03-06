# -*- coding: utf-8 -*-
from __future__ import absolute_import
import functools
import itertools
import math
from operator import mul
import six
from six.moves import reduce
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
        value = func(*args, **kwargs)
        if value is None:
            return {}
        return {key: value}
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


@single_value
def suffix3(token, parses):
    return token.lower()[-3:]


@single_value
def suffix2(token, parses):
    return token.lower()[-2:]


class Drop(object):
    def __init__(self, key):
        self.key = key

    def __call__(self, tokens, parsed_tokens, feature_dicts):
        for featdict in feature_dicts:
            if self.key in featdict:
                del featdict[self.key]


class _GrammemeFeatures(object):
    default_name = None
    default_threshold = 0.0

    # TODO: weighting: max / sum / one-zero / ...?
    def __init__(self, name=None, threshold=None, add_unambig=False, ignore=None, only=None):
        self.name = name if name is not None else self.default_name
        self.unambig_name = self.name + '[unambig]'
        self.threshold = threshold if threshold is not None else self.default_threshold
        self.add_unambig = add_unambig
        self.ignore = set(ignore) if ignore is not None else set()
        self.only = set(only) if only is not None else None

    def __call__(self, token, parses):
        parses = [p for p in parses if p.score >= self.threshold]
        return self.extract(parses)

    def _filtered_grammemes(self, parse):
        grammemes = [gr for gr in parse.tag._grammemes_tuple if gr not in self.ignore]
        if self.only is not None:
            grammemes = [gr for gr in grammemes if gr in self.only]
            if not grammemes:
                grammemes = ['NA']
        return grammemes

    def extract(self, parses):
        raise NotImplementedError()


class Grammeme(_GrammemeFeatures):
    """
    This feature adds all seen grammemes with their weights to the result.
    If there are several weights possible, the maximum is used.
    """
    default_name = 'Grammeme'

    def extract(self, parses):
        features = {}
        features_unambig = {}

        for p in parses:
            for grammeme in self._filtered_grammemes(p):
                # TODO/FIXME: sum instead of max or in addition to max
                features[grammeme] = max(p.score, features.get(grammeme, 0))
                # features[grammeme] = features.get(grammeme, 0) + p.score

                # TODO/FIXME: grammeme is unambiguous when its scores sums to 1?
                if self.add_unambig and p.score == 1:
                    features_unambig[grammeme] = 1

        res = {self.name: features}
        if self.add_unambig:
            res[self.unambig_name] = features_unambig
        return res


class GrammemePair(_GrammemeFeatures):
    """
    This feature adds all seen grammeme pairs with their weights to the result.
    If there are several weights possible, the maximum is used.
    """
    default_name = 'GrammemePair'
    default_threshold = 0.1

    def extract(self, parses):
        features = {}
        features_unambig = {}
        for p in parses:
            grammemes = self._filtered_grammemes(p)
            for pair in _iter_grammeme_pairs(grammemes):
                # TODO/FIXME: sum instead of max or in addition to max
                features[pair] = max(p.score, features.get(pair, 0))
                # features[pair] = features.get(pair, 0) + p.score

                # TODO/FIXME: grammeme is unambiguous when its scores sums to 1
                if self.add_unambig and p.score == 1:
                    features_unambig[pair] = 1

        res = {self.name: features}
        if self.add_unambig:
            res[self.unambig_name] = features_unambig
        return res


def _iter_grammeme_pairs(grammemes):
    for idx, grammeme in enumerate(grammemes):
        for grammeme2 in grammemes[idx+1:]:
            # make grammeme order always the same
            if grammeme < grammeme2:
                yield ",".join([grammeme, grammeme2])
            else:
                yield ",".join([grammeme2, grammeme])



class Pattern(object):
    """
    Global feature that combines local features.
    At each position it evaluate `*patterns` passed to constructor and
    updates feature dicts accordingly.

    Example::

        >>> from pymorphy2 import MorphAnalyzer
        >>> from morphine import features
        >>> from morphine.feature_extractor import FeatureExtractor
        >>> morph = MorphAnalyzer()
        >>> fe = FeatureExtractor(
        ...     token_features=[features.token_lower],
        ...     global_features=[
        ...         Pattern([-1, 'token_lower']),  # previous token
        ...         Pattern([-1, 'token_lower'], [0, 'token_lower']),  # a bigram
        ...     ]
        ... )
        >>> sent = ['Hello', 'my', 'darling']
        >>> parsed = [morph.parse(word) for word in sent]
        >>> xseq = fe.transform_single(sent, parsed)
        >>> len(sent) == len(xseq)
        True
        >>> xseq[2] == {
        ...    'token_lower': 'darling',
        ...    'token_lower[i-1]/token_lower[i]': 'my/darling',
        ...    'token_lower[i-1]': 'my'
        ... }
        True


    Parameters
    ----------

    *patterns : tuples
        :class:`Pattern` accept one or more patterns as positional parameters.
        Each pattern should be a tuple with 2 or 3 elements: either
        ``(offset, feature)`` or ``(offset, feature, name)``.

        ``offset`` is an integer; for each position *i* feature values
        for *i+offset* are calculated.

        ``name`` is a name of this feature. If missing, it is auto-generated
        from ``offset`` and ``feature``.

        ``feature`` could be either a callable or a string.
        If ``feature`` is a string then it is used as a lookup key in
        the current feature dict. When ``feature`` is a callable it is called
        to get feature value.

        Callable ``feature`` can accept either 2 or 3 arguments:
        either ``token`` and ``parses`` or ``token``, ``parses`` and
        ``feature_dict``. ``token`` is a current token (as text);
        ``parses`` is a list of token parses as returned by pymorphy2;
        ``feature_dict`` is the current feature dict.

        Callable features should return the value; they shouldn't update
        ``feature_dict`` themselves.

    name : str, optional
        Name of this pattern (a key in feature dict to update).
        If missing, it is auto-generated from the passed patterns.

    index_low : integer, optional
        Minimum index to evaluate patterns at. By default, patterns
        are evaluated only at positions where all values are present.
        I.e. if you have offset==-2 in the pattern, patterns won't be evaluated
        at positions 0 and 1. You can override that by passing ``index_low``
        keyword argument. Values that are not available will be replaced by
        ``?`` signs.

    index_high : integer, optional
        Maximum index to evaluate patterns at. See :param:`index_low`.

    """
    separator = '/'
    out_value = '?'
    missing_value = 0.0

    def __init__(self, *patterns, **kwargs):
        # save original arguments to make pickling/unpickling easier
        self._init_patterns = patterns
        self._init_kwargs = kwargs
        self._init()

    def _init(self):
        self.patterns = []
        index_low, index_high, names = 0, 0, []

        for pattern in self._init_patterns:
            offset, feat, name = self._parse_pattern(pattern)
            func = self._get_feature_func(feat)
            self.patterns.append((offset, func, name))

            if index_low < -offset:
                index_low = -offset
            if index_high < offset:
                index_high = offset
            names.append(name)

        self.index_low = self._init_kwargs.get('index_low', index_low)
        self.index_high = self._init_kwargs.get('index_high', index_high)
        self.name = self._init_kwargs.get('name', self.separator.join(names))

        if len(self.patterns) == 1:
            self._get_combined_value = self._get_combined_value_single
        else:
            self._get_combined_value = self._get_combined_value_multi

    def _parse_pattern(self, pattern):
        if len(pattern) == 2:
            offset, feat = pattern
            self._validate_offset(offset)
            return offset, feat, self._auto_feature_name(offset, feat)
        elif len(pattern) == 3:
            offset, feat, name = pattern
            self._validate_offset(offset)
            return offset, feat, name
        else:
            raise ValueError("Patterns must be tuples with 2 or 3 elements")

    def _auto_feature_name(self, offset, feat):
        func_name = feat.__name__ if callable(feat) else feat
        if offset == 0:
            return func_name + "[i]"
        elif offset < 0:
            return "%s[i%s]" % (func_name, offset)
        else:
            return "%s[i+%s]" % (func_name, offset)

    def _validate_offset(self, offset):
        if not isinstance(offset, int):
            raise ValueError("Offset must be integer")

    @classmethod
    def _add_feat_dict_argument_if_missing(cls, func):
        if func_takes_argument(func, 'feature_dict'):
            return func

        @functools.wraps(func)
        def _func(token, parses, feature_dict):
            return func(token, parses)
        return _func

    def _get_feature_func(self, feat):
        if callable(feat):  # "standard" token features
            return self._add_feat_dict_argument_if_missing(feat)
        else:  # dictionary lookup
            return self._get_lookup_func(key=feat)

    def _get_lookup_func(self, key):
        def lookup(token, parses, feature_dict, key=key):
            return feature_dict.get(key, self.missing_value)
        return lookup

    def __call__(self, tokens, parsed_tokens, feature_dicts):
        sliced = feature_dicts[self.index_low:]
        if self.index_high:
            sliced = sliced[:-self.index_high]

        for pos, featdict in enumerate(sliced, start=self.index_low):
            values = []
            for offset, func, name in self.patterns:
                index = pos + offset
                if 0 <= index < len(feature_dicts):
                    value = func(tokens[index], parsed_tokens[index], feature_dicts[index])
                else:
                    value = self.out_value
                values.append(value)

            featdict[self.name] = self._get_combined_value(values)

    def _get_combined_value_single(self, values):
        return values[0]

    def _get_combined_value_multi(self, values):
        if all(isinstance(v, dict) for v in values):
            # Cartesian product of all keys; values are multiplied.
            combined_value = {}
            for items in itertools.product(*(v.items() for v in values)):
                _keys, _values = zip(*items)
                name = self.separator.join(_keys)
                value = reduce(mul, map(float, _values))
                combined_value[name] = value
            return combined_value

        elif len(values) == 2 and isinstance(values[1], dict):
            # (single feature, dict feature)
            return {values[0]: values[1]}

        elif len(values) == 2 and isinstance(values[0], dict):
            # (dict feature, single feature)
            return {values[1]: values[0]}

        else:
            for value in values:
                if isinstance(value, float) and value not in {0.0, 1.0}:
                    raise ValueError("Values must be boolean or string for Pattern to work")
            return self.separator.join(map(six.text_type, values))

    def __getstate__(self):
        return {
            '_init_patterns': self._init_patterns,
            '_init_kwargs': self._init_kwargs,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init()
