# -*- coding: utf-8 -*-
from __future__ import absolute_import
from morphine import features
from morphine.feature_extractor import FeatureExtractor


def test_pattern(morph):
    fe = FeatureExtractor(
        [features.token_lower],
        [
            features.sentence_start,
            features.sentence_end,
            features.Pattern([-1, 'token_lower']),
        ],
    )
    sent = 'Летят гуси на юг'.split()
    parsed = [morph.parse(t) for t in sent]
    assert fe.transform_single(sent, parsed) == [
        {'token_lower': 'летят', 'sentence_start': 1.0},
        {'token_lower': 'гуси', 'token_lower[i-1]': 'летят'},
        {'token_lower': 'на', 'token_lower[i-1]': 'гуси'},
        {'token_lower': 'юг', 'sentence_end': 1.0, 'token_lower[i-1]': 'на'},
    ]


def test_pattern_bigram(morph):
    sent = 'Летят гуси на юг'.split()
    parsed = [morph.parse(t) for t in sent]
    fe = FeatureExtractor(
        [features.token_lower],
        [
            features.sentence_start,
            features.sentence_end,
            features.Pattern([-1, 'token_lower'], [-1, 'sentence_start']),
        ],
    )
    assert fe.transform_single(sent, parsed) == [
        {'token_lower': 'летят', 'sentence_start': 1.0},
        {'token_lower': 'гуси', 'token_lower[i-1]/sentence_start[i-1]': 'летят/1.0'},
        {'token_lower': 'на', 'token_lower[i-1]/sentence_start[i-1]': 'гуси/0.0'},
        {'token_lower': 'юг', 'sentence_end': 1.0, 'token_lower[i-1]/sentence_start[i-1]': 'на/0.0'},
    ]


def test_pattern_cartesian(morph):
    sent = 'Летят гуси на юг'.split()
    parsed = [morph.parse(t) for t in sent]
    fe = FeatureExtractor(
        [features.token_lower, features.Grammeme(threshold=0.1)],
        [
            features.Pattern([-1, 'Grammeme'], [0, 'Grammeme']),
            features.Drop('Grammeme')
        ],
    )
    xseq = fe.transform_single(sent, parsed)
    assert xseq[0] == {'token_lower': 'летят'}
    assert sorted(xseq[1].keys()) == sorted(['Grammeme[i-1]/Grammeme[i]', 'token_lower'])
    assert xseq[1]['Grammeme[i-1]/Grammeme[i]']['VERB/NOUN'] == 1.0


def test_pattern_bigram_with_dict(morph):
    sent = 'Летят гуси на юг'.split()
    parsed = [morph.parse(t) for t in sent]
    fe = FeatureExtractor(
        [features.token_lower, features.Grammeme(threshold=0.1)],
        [
            features.Pattern([-1, 'Grammeme'], [0, 'token_lower']),
            features.Pattern([-1, 'token_lower'], [0, 'Grammeme']),
        ],
    )
    xseq = fe.transform_single(sent, parsed)
    assert sorted(xseq[1].keys()) == sorted([
        'Grammeme',
        'Grammeme[i-1]/token_lower[i]',
        'token_lower',
        'token_lower[i-1]/Grammeme[i]',
    ])
    assert xseq[1]['Grammeme[i-1]/token_lower[i]'] == {'гуси': xseq[0]['Grammeme']}
    assert xseq[1]['token_lower[i-1]/Grammeme[i]'] == {'летят': xseq[1]['Grammeme']}


def test_pattern_kwargs(morph):
    sent = 'Летят гуси на юг'.split()
    parsed = [morph.parse(t) for t in sent]
    fe = FeatureExtractor(
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
    assert fe.transform_single(sent, parsed) == [
        {'token_lower': 'летят', 'sentence_start': 1.0, 'low+1 BOS-1': 'гуси/?'},
        {'token_lower': 'гуси', 'low+1 BOS-1': 'на/1.0'},
        {'token_lower': 'на', 'low+1 BOS-1': 'юг/0.0'},
        {'token_lower': 'юг', 'sentence_end': 1.0, 'low+1 BOS-1': '?/0.0'},
    ]


def test_pattern_callable(morph):
    sent = 'Летят гуси на юг'.split()
    parsed = [morph.parse(t) for t in sent]
    fe = FeatureExtractor([], [
        features.Pattern(
            [0, lambda token, parses: token.istitle(), 'title'],
        ),
    ])
    assert fe.transform_single(sent, parsed) == [
        {'title': True},
        {'title': False},
        {'title': False},
        {'title': False},
    ]


def test_pattern_callable_complex(morph):
    sent = 'Летят гуси на юг'.split()
    parsed = [morph.parse(t) for t in sent]

    def not_title(token, parses, feature_dict):
        return not feature_dict.get('title', False)

    fe = FeatureExtractor([], [
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
    assert fe.transform_single(sent, parsed) == [
        {'title': True},
        {'title': False, 'title[i-1]/title[i]': 'True/False', 'not_title[i-1]/not_title[i]/not_title[i+1]': 'False/True/True'},
        {'title': False, 'title[i-1]/title[i]': 'False/False', 'not_title[i-1]/not_title[i]/not_title[i+1]': 'True/True/True'},
        {'title': False, 'title[i-1]/title[i]': 'False/False'},
    ]


def test_pattern_names():
    assert features.Pattern([0, 'foo']).name == 'foo[i]'
    assert features.Pattern([-1, 'foo'], [2, 'bar']).name == 'foo[i-1]/bar[i+2]'

    def baz(token, parses):
        pass

    assert features.Pattern([0, baz]).name == 'baz[i]'
    assert features.Pattern([0, baz], [1, 'spam']).name == 'baz[i]/spam[i+1]'
    assert features.Pattern([1, 'spam'], [0, baz]).name == 'spam[i+1]/baz[i]'

    assert features.Pattern([0, baz, 'egg']).name == 'egg'

    assert features.Pattern([0, baz], name='fuzz').name == 'fuzz'
    assert features.Pattern([0, baz, 'egg'], name='fuzz').name == 'fuzz'


def test_pattern_low_high():
    assert features.Pattern([-2,'foo']).index_low == 2
    assert features.Pattern([-2,'foo']).index_high == 0
    assert features.Pattern([-2,'foo'], index_low=5).index_low == 5
    assert features.Pattern([-2,'foo'], [1, 'bar']).index_high == 1
    assert features.Pattern([-2,'foo'], [1, 'bar'], index_high=2).index_high == 2
