# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from morphine import features
from morphine.feature_extractor import FeatureExtractor


def test_token_features(morph):
    fe = FeatureExtractor(morph,
        [features.bias, features.token_lower],
    )
    assert fe.transform_single('Летят гуси'.split()) == [
        {'bias': 1, 'token_lower': 'летят'},
        {'bias': 1, 'token_lower': 'гуси'},
    ]


def test_token_and_global_features(morph):
    sent = 'Летят гуси на юг'.split()
    fe = FeatureExtractor(morph,
        [features.token_lower],
        [features.sentence_start, features.sentence_end],
    )
    assert fe.transform_single(sent) == [
        {'token_lower': 'летят', 'sentence_start': 1.0},
        {'token_lower': 'гуси'},
        {'token_lower': 'на'},
        {'token_lower': 'юг', 'sentence_end': 1.0},
    ]

    assert fe.transform_single('юг'.split()) == [
        {'token_lower': 'юг', 'sentence_start': 1.0, 'sentence_end': 1.0},
    ]

    assert [fe.transform_single(sent)] == fe.transform([sent])
    assert [fe.transform_single(sent)] == fe.fit_transform([sent])
