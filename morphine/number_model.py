# -*- coding: utf-8 -*-
from __future__ import absolute_import

from morphine import features
from morphine.feature_extractor import FeatureExtractor
from morphine.basetagger import PartialTagger


class NumberFeatureExtractor(FeatureExtractor):
    IGNORE = {'Arch', 'intg', 'real', 'GNdr', 'Ms-f'}

    def __init__(self):
        super(NumberFeatureExtractor, self).__init__(
            token_features=[
                features.bias,
                features.token_lower,
                features.Grammeme(threshold=0.01, add_unambig=True, ignore=self.IGNORE),
                features.GrammemePair(threshold=0.0, add_unambig=True, ignore=self.IGNORE),
            ],
            global_features=[
                # features.sentence_start,
                # features.sentence_end,

                # features.Pattern([-1, 'token_lower']),
                # features.Pattern([-2, 'token_lower']),

                features.Pattern([-1, 'Grammeme']),
                features.Pattern([+1, 'Grammeme']),

                # features.Pattern([-2, 'Grammeme'], [-1, 'Grammeme']),
                features.Pattern([-1, 'Grammeme'], [0, 'Grammeme']),
                features.Pattern([-1, 'Grammeme'], [0, 'GrammemePair']),

                features.Pattern([-1, 'GrammemePair']),
                features.Pattern([+1, 'GrammemePair']),
            ],
        )


class Tagger(PartialTagger):
    def outval(self, tag):
        return str(tag.number or 'NA')
