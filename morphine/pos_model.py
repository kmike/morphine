# -*- coding: utf-8 -*-
from __future__ import absolute_import
from pymorphy2 import MorphAnalyzer
from morphine import features
from morphine.feature_extractor import FeatureExtractor


class POSFeatureExtractor(FeatureExtractor):
    IGNORE = {'Arch', 'intg', 'real', '1per', '2per', '3per', 'GNdr', 'Ms-f'}

    def __init__(self, morph=None):
        morph = morph or MorphAnalyzer()

        super(POSFeatureExtractor, self).__init__(
            morph=morph,
            token_features=[
                features.bias,
                features.token_lower,
                features.Grammeme(threshold=0.01, add_unambig=True, ignore=self.IGNORE),
                features.GrammemePair(threshold=0.0, add_unambig=True, ignore=self.IGNORE),
            ],
            global_features=[
                features.sentence_start,
                features.sentence_end,

                features.Pattern([-1, 'token_lower']),
                features.Pattern([+1, 'token_lower']),

                features.Pattern([-1, 'Grammeme']),
                features.Pattern([+1, 'Grammeme']),

                features.Pattern([-1, 'GrammemePair']),
                features.Pattern([+1, 'GrammemePair']),
            ],
        )
