# -*- coding: utf-8 -*-
from __future__ import absolute_import

from morphine import features
from morphine.feature_extractor import FeatureExtractor
from morphine.basetagger import PartialTagger

from pymorphy2.tagset import OpencorporaTag

class POSFeatureExtractor(FeatureExtractor):
    IGNORE = {
        'Arch', 'intg', 'real', '1per', '2per', '3per', 'GNdr', 'Ms-f',
        'anim', 'inan',
        'masc', 'femn', 'neut',
        'Geox', 'Name',
    } | OpencorporaTag.CASES | OpencorporaTag.NUMBERS | OpencorporaTag.MOODS \
      | OpencorporaTag.INVOLVEMENT

    def __init__(self):
        super(POSFeatureExtractor, self).__init__(
            token_features=[
                features.bias,
                features.token_lower,
                features.suffix2,
                features.suffix3,
                features.Grammeme(threshold=0.01, add_unambig=False, ignore=self.IGNORE),
                features.GrammemePair(threshold=0.01**2, add_unambig=False, ignore=self.IGNORE),
            ],
            global_features=[
                features.sentence_start,
                features.sentence_end,

                # features.the_only_verb,

                features.Pattern([-1, 'token_lower']),
                # features.Pattern([+1, 'token_lower']),

                features.Pattern([-1, 'Grammeme']),
                features.Pattern([+1, 'Grammeme']),

                features.Pattern([-1, 'GrammemePair']),
                features.Pattern([+1, 'GrammemePair']),

                # features.Pattern([-1, 'GrammemePair'], [0, 'GrammemePair']),
            ],
        )


class Tagger(PartialTagger):
    def outval(self, tag):
        return tag._POS
