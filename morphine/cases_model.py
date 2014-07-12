# -*- coding: utf-8 -*-
from __future__ import absolute_import
from collections import defaultdict

import pycrfsuite

from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize


class CaseFeatureExtractor(object):

    THRESH = 0.1

    def __init__(self, morph):
        self.morph = morph

    def _add_grammeme_features(self, token, namespace, features, k=1):
        for p in self.morph.parse(token):
            if p.score < self.THRESH:
                continue

            for grammeme in p.tag.grammemes:
                # add all grammemes
                key = "%s:%s" % (namespace, grammeme)
                features[key] = max(p.score*k, features[key])
                if p.score == 1:
                    features['unambig:'+key] = 1

                # add all combinations of 2 grammemes
                seen = set()
                for grammeme2 in p.tag.grammemes:
                    if grammeme2 == grammeme:
                        continue

                    if grammeme > grammeme2:
                        key2 = "%s:%s,%s" % (namespace, grammeme, grammeme2)
                    else:
                        key2 = "%s:%s,%s" % (namespace, grammeme2, grammeme)

                    if key2 in seen:
                        continue
                    seen.add(key2)
                    features[key2] = max(p.score*k, features[key2])

                    if p.score == 1:
                        features['unambig:'+key2] = 1

    def _get_features(self, tokens, i):
        token = tokens[i]
        features = defaultdict(float)
        features['bias'] = 1

        features['i:token:%s' % token.lower()] = 1
        self._add_grammeme_features(token, "i", features, k=2)

        if i > 0:
            features['i-1:token:%s' % tokens[i-1].lower()] = 1
            self._add_grammeme_features(tokens[i-1], "i-1", features)
        else:
            features['BOS'] = 1  # begin-of-sentence mark

        if i > 1:
            features['i-2:token:%s' % tokens[i-2].lower()] = 1
            self._add_grammeme_features(tokens[i-2], "i-2", features)

        if i < len(tokens) - 1:
            features['i+1:token:%s' % tokens[i+1].lower()] = 1
            self._add_grammeme_features(tokens[i+1], "i+1", features)
        else:
            features['EOS'] = 1


        return dict(features)

    def transform_sent(self, tokens):
        return [self._get_features(tokens, i) for i in range(len(tokens))]


class Tagger(object):

    def __init__(self, model_filename, morph=None, feature_extractor=None,
                 model_strips_pnct=True):
        if morph is None:
            morph = MorphAnalyzer()
        self.morph = morph

        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(model_filename)

        if feature_extractor is None:
            feature_extractor = CaseFeatureExtractor(morph)
        self.feature_extractor = feature_extractor

        self.strip_pnct = model_strips_pnct

    def predict(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            tokens = simple_word_tokenize(tokens)

        pymorphy_res = [self.morph.parse(tok) for tok in tokens]

        if self.strip_pnct:
            tokens = [token for token, parses in zip(tokens, pymorphy_res)
                      if 'PNCT' not in parses[0].tag]

        cases = self.tagger.tag(self.feature_extractor.transform_sent(tokens))

        i = 0
        res = []
        for parses in pymorphy_res:
            if self.strip_pnct and 'PNCT' in parses[0].tag:
                res.append(parses[0])
                continue

            case = cases[i]

            def keyfunc(p):
                if case == 'NA':
                    match = p.tag.case is None
                else:
                    match = case in self.morph.TagClass.fix_rare_cases(p.tag.grammemes)

                return match, p.score

            p = max(parses, key=keyfunc)
            if case != 'NA':
                p = p._replace(score=self.tagger.marginal(case, i))
            res.append(p)
            i += 1

        return res
