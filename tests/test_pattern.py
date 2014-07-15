# -*- coding: utf-8 -*-
from __future__ import absolute_import
from morphine import features
from morphine.feature_extractor import FeatureExtractor


def test_pattern(morph):
    fe = FeatureExtractor(morph,
        [features.token_lower],
        [
            features.sentence_start,
            features.sentence_end,
            features.Pattern([-1, 'token_lower']),
        ],
    )
    sent = 'Летят гуси на юг'.split()
    assert fe.transform_single(sent) == [
        {'token_lower': 'летят', 'sentence_start': 1.0},
        {'token_lower': 'гуси', 'token_lower[-1]': 'летят'},
        {'token_lower': 'на', 'token_lower[-1]': 'гуси'},
        {'token_lower': 'юг', 'sentence_end': 1.0, 'token_lower[-1]': 'на'},
    ]


def test_pattern2(morph):
    sent = 'Летят гуси на юг'.split()
    fe = FeatureExtractor(morph,
        [features.token_lower],
        [
            features.sentence_start,
            features.sentence_end,
            features.Pattern([-1, 'token_lower'], [-1, 'sentence_start']),
        ],
    )
    assert fe.transform_single(sent) == [
        {'token_lower': 'летят', 'sentence_start': 1.0},
        {'token_lower': 'гуси', 'token_lower[-1]/sentence_start[-1]': 'летят/1.0'},
        {'token_lower': 'на', 'token_lower[-1]/sentence_start[-1]': 'гуси/0.0'},
        {'token_lower': 'юг', 'sentence_end': 1.0, 'token_lower[-1]/sentence_start[-1]': 'на/0.0'},
    ]


def test_pattern_kwargs(morph):
    sent = 'Летят гуси на юг'.split()
    fe = FeatureExtractor(morph,
        [features.token_lower],
        [
            features.sentence_start,
            features.sentence_end,
            features.Pattern(
                [+1, 'token_lower'],
                [-1, 'sentence_start'],
                name='low+1 BOS-1',
                index_low=0,
                index_high=0,
            ),
        ],
    )
    assert fe.transform_single(sent) == [
        {'token_lower': 'летят', 'sentence_start': 1.0, 'low+1 BOS-1': 'гуси/?'},
        {'token_lower': 'гуси', 'low+1 BOS-1': 'на/1.0'},
        {'token_lower': 'на', 'low+1 BOS-1': 'юг/0.0'},
        {'token_lower': 'юг', 'sentence_end': 1.0, 'low+1 BOS-1': '?/0.0'},
    ]


def test_pattern_callable(morph):
    sent = 'Летят гуси на юг'.split()
    fe = FeatureExtractor(morph, [], [
        features.Pattern(
            [0, lambda token, parses: token.istitle(), 'title'],
        ),
    ])
    assert fe.transform_single(sent) == [
        {'title': True},
        {'title': False},
        {'title': False},
        {'title': False},
    ]


def test_pattern_callable_complex(morph):
    sent = 'Летят гуси на юг'.split()

    def not_title(token, parses, feature_dict):
        return not feature_dict.get('title', False)

    fe = FeatureExtractor(morph, [], [
        features.Pattern(
            [0, lambda token, parses: token.istitle(), 'title'],
        ),
        features.Pattern(
            [-1, 'title'],
            [ 0, 'title'],
        ),
        features.Pattern(
            [-1, not_title],
            [ 0, not_title],
            [+1, not_title],
        )
    ])
    assert fe.transform_single(sent) == [
        {'title': True},
        {'title': False, 'title[-1]/title[0]': 'True/False', 'not_title[-1]/not_title[0]/not_title[+1]': 'False/True/True'},
        {'title': False, 'title[-1]/title[0]': 'False/False', 'not_title[-1]/not_title[0]/not_title[+1]': 'True/True/True'},
        {'title': False, 'title[-1]/title[0]': 'False/False'},
    ]


def test_pattern_names():
    assert features.Pattern([0, 'foo']).name == 'foo[0]'
    assert features.Pattern([-1, 'foo'], [2, 'bar']).name == 'foo[-1]/bar[+2]'

    def baz(token, parses):
        pass

    assert features.Pattern([0, baz]).name == 'baz[0]'
    assert features.Pattern([0, baz], [1, 'spam']).name == 'baz[0]/spam[+1]'
    assert features.Pattern( [1, 'spam'], [0, baz]).name == 'spam[+1]/baz[0]'

    assert features.Pattern([0, baz, 'egg']).name == 'egg'

    assert features.Pattern([0, baz], name='fuzz').name == 'fuzz'
    assert features.Pattern([0, baz, 'egg'], name='fuzz').name == 'fuzz'


def test_pattern_low_high():
    assert features.Pattern([-2,'foo']).index_low == 2
    assert features.Pattern([-2,'foo']).index_high == 0
    assert features.Pattern([-2,'foo'], index_low=5).index_low == 5
    assert features.Pattern([-2,'foo'], [1, 'bar']).index_high == 1
    assert features.Pattern([-2,'foo'], [1, 'bar'], index_high=2).index_high == 2
