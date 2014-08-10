# -*- coding: utf-8 -*-
from __future__ import absolute_import
try:
    from cytoolz import functoolz, dicttoolz
except ImportError:
    from toolz import functoolz, dicttoolz


class FeatureExtractor(object):
    """
    This class extracts features from sentences (lists of tokens).

    :meth:`fit` / :meth:`transform` / :meth:`fit_transform` interface
    may look familiar to you if you ever used scikit-learn_.
    But there is one twist: for POS tagging task (and other sequence
    labelling tasks) a single observation is a sentence, not an individual word.
    So :meth:`fit` / :meth:`transform` / :meth:`fit_transform` methods accept
    lists of sentences (lists of lists of token information), and return lists
    of sentences' feature dicts (lists of lists of feature dicts).

    .. _scikit-learn: http://scikit-learn.org

    Parameters
    ----------

    morph : pymorphy2.MorphAnalyzer
        MorphAnalyzer instance.

    token_features : list of callables
        List of "token" feature functions. Each function accepts
        two arguments: ``token`` and ``parses``, and returns a dictionary
        wich maps feature names to feature values.

        ``token`` is the token text; ``parses`` is a result of parsing this
        token by pymorphy2.

        Dicts from all token feature functions are merged by FeatureExtractor.
        Example token feature (it just returns token text)::

            >>> def current_token_lower(token, parses):
            ...     return {'token_lower': token.lower()}

    global_features : list of callables, optional
        List of "global" feature functions. Each "global" feature function
        accepts 3 arguments:

        * ``tokens`` - a list of tokens in this sentence;
        * ``parsed_tokens`` - a list of pymorphy2 parses for all tokens
          in a sentence;
        * ``feature_dicts`` - feature dicts for each token (extracted by
          token feature functions and modified by previous global feature
          functions). Global feature functions should work by changing
          ``feature_dict`` inplace.

        Global feature functions are applied after token feature
        functions in the order they are passed.
    """
    def __init__(self, morph, token_features, global_features=None):
        self.morph = morph
        self.combined_token_features = _CombinedFeatures(*token_features)
        self.global_features = global_features or []

    def fit(self, token_lists, y=None):
        self.fit_transform(token_lists)
        return self

    def fit_transform(self, token_lists, y=None, **fit_params):
        return self.transform(token_lists)

    def transform(self, token_lists):
        return [self.transform_single(tokens) for tokens in token_lists]

    def transform_single(self, tokens):
        parsed_tokens = [self.morph.parse(tok) for tok in tokens]
        feature_dicts = list(map(self.combined_token_features, tokens, parsed_tokens))

        for feat in self.global_features:
            feat(tokens, parsed_tokens, feature_dicts)

        return feature_dicts


class _CombinedFeatures(object):
    """
    Utility for combining several feature functions::

        >>> from pprint import pprint
        >>> def f1(tok): return {'upper': tok.isupper()}
        >>> def f2(tok): return {'len': len(tok)}
        >>> features = _CombinedFeatures(f1, f2)
        >>> pprint(features('foo'))
        {'len': 3, 'upper': False}

    """
    def __init__(self, *feature_funcs):
        self._combined = functoolz.juxt(feature_funcs)

    def __call__(self, *args, **kwargs):
        return dicttoolz.merge(*self._combined(*args, **kwargs))
