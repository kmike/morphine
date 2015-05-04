# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from morphine import features
from morphine.feature_extractor import FeatureExtractor


def test_token_features(morph):
    fe = FeatureExtractor(
        [features.bias, features.token_lower],
    )
    tokens = 'Летят гуси'.split()
    parsed = [morph.parse(t) for t in tokens]
    assert fe.transform_single(tokens, parsed) == [
        {'bias': 1, 'token_lower': 'летят'},
        {'bias': 1, 'token_lower': 'гуси'},
    ]


def test_token_and_global_features(morph):
    sent = 'Летят гуси на юг'.split()
    parsed = [morph.parse(t) for t in sent]
    fe = FeatureExtractor(
        [features.token_lower],
        [features.sentence_start, features.sentence_end],
    )
    assert fe.transform_single(sent, parsed) == [
        {'token_lower': 'летят', 'sentence_start': 1.0},
        {'token_lower': 'гуси'},
        {'token_lower': 'на'},
        {'token_lower': 'юг', 'sentence_end': 1.0},
    ]

    assert [fe.transform_single(sent, parsed)] == fe.transform(zip([sent], [parsed]))
    assert [fe.transform_single(sent, parsed)] == fe.fit_transform(zip([sent], [parsed]))

    sent = 'юг'.split()
    parsed = [morph.parse(t) for t in sent]
    assert fe.transform_single(sent, parsed) == [
        {'token_lower': 'юг', 'sentence_start': 1.0, 'sentence_end': 1.0},
    ]

