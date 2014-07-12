# -*- coding: utf-8 -*-
from __future__ import absolute_import


class FeatureExtractor(object):
    """
    This class extracts features from sentences. Each sentence should be a
    list of ``(token, parses)`` tuples.

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

    token_features : list of callables
        List of "token" feature functions. Each function accepts
        two arguments: ``token`` and ``parses``, and returns a dictionary
        wich maps feature names to feature values. Dicts from all
        token feature functions are merged by HtmlFeatureExtractor.
        Example token feature (it just returns token text)::

            >>> def current_token_lower(token, parses):
            ...     return {'token_lower': token.lower()}

    global_features : list of callables, optional
        List of "global" feature functions. Each "global" feature function
        should accept a single argument - a list
        of ``(token, parses, feature_dict)`` tuples.
        This list contains all tokens from the document and
        features extracted by previous feature functions.

        "Global" feature functions are applied after "token" feature
        functions in the order they are passed.

        They should change feature dicts ``feature_dict`` inplace.

    min_df : integer or Mapping, optional
        Feature values that have a document frequency strictly
        lower than the given threshold are removed.
        If ``min_df`` is integer, its value is used as threshold.

        TODO: if ``min_df`` is a dictionary, it should map feature names
        to thresholds.

    """
    def __init__(self, token_features, global_features=None, min_df=1):
        self.token_features = token_features
        self.global_features = global_features or []
        self.min_df = min_df

    def fit(self, token_lists, y=None):
        self.fit_transform(token_lists)
        return self

    def fit_transform(self, token_lists, y=None, **fit_params):
        X = [self.transform_single(tokens) for tokens in token_lists]
        return self._pruned(X, low=self.min_df)

    def transform(self, html_token_lists):
        return [self.transform_single(html_tokens) for html_tokens in html_token_lists]

    def transform_single(self, html_tokens):
        feature_func = _CombinedFeatures(*self.token_features)
        token_data = list(zip(html_tokens, map(feature_func, html_tokens)))

        for feat in self.global_features:
            feat(token_data)

        return [featdict for tok, featdict in token_data]

    def _pruned(self, X, low=None):
        if low is None or low <= 1:
            return X
        cnt = self._document_frequency(X)
        keep = {k for (k, v) in cnt.items() if v >= low}
        del cnt
        return [
            [{k: v for k, v in fd.items() if (k, v) in keep} for fd in doc]
            for doc in X
        ]

    def _document_frequency(self, X):
        cnt = Counter()
        for doc in X:
            seen_features = set(chain.from_iterable(fd.items() for fd in doc))
            cnt.update(seen_features)
        return cnt


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
