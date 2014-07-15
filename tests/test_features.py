# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from morphine import features

def _parse(tokens, morph):
    return [morph.parse(tok) for tok in tokens]


def test_bias(morph):
    assert features.bias([], []) == {'bias': 1}
    assert features.bias(['Привет'], morph.parse('Привет')) == {'bias': 1}


def test_token_lower(morph):
    assert features.token_lower('ПрИвет', morph.parse('ПрИвет')) == {'token_lower': 'привет'}


def test_token_identity(morph):
    tok = 'ПрИвет'
    assert features.token_identity(tok, morph.parse(tok)) == {'token_identity': tok}


def test_sentence_start(morph):
    tokens = 'Гуси летят на юг'.split()
    parses = _parse(tokens, morph)
    feature_dicts = [{} for tok in tokens]

    features.sentence_start(tokens, parses, feature_dicts)
    assert feature_dicts[0] == {'sentence_start': 1.0}
    assert feature_dicts[1:] == [{} for tok in tokens[1:]]

    feature_dicts = []
    features.sentence_start([], [], feature_dicts)
    assert feature_dicts == []


def test_sentence_end(morph):
    tokens = 'Гуси летят на юг'.split()
    parses = _parse(tokens, morph)
    feature_dicts = [{} for tok in tokens]
    features.sentence_end(tokens, parses, feature_dicts)
    assert feature_dicts[-1] == {'sentence_end': 1.0}
    assert feature_dicts[:-1] == [{} for tok in tokens[:-1]]

    feature_dicts = []
    features.sentence_end([], [], feature_dicts)
    assert feature_dicts == []


def test_Grammeme(morph):
    feat = features.Grammeme()
    res = feat('на', morph.parse('на'))
    assert sorted(res.keys()) == ['Grammeme:INTJ', 'Grammeme:PRCL', 'Grammeme:PREP']
    assert res['Grammeme:PREP'] > res['Grammeme:PRCL']
    assert res['Grammeme:PREP'] > res['Grammeme:INTJ']

    res = feat('стали', morph.parse('стали'))
    assert 'Grammeme:past' in res
    assert 'Grammeme:accs' in res
    assert res['Grammeme:VERB'] > res['Grammeme:NOUN']


def test_Grammeme_threshold(morph):
    feat = features.Grammeme(threshold=0.1)
    res = feat('на', morph.parse('на'))
    assert sorted(res.keys()) == ['Grammeme:PREP']
    assert res['Grammeme:PREP'] > 0.99


def test_GrammemePair(morph):
    feat = features.GrammemePair()
    res = feat('на', morph.parse('на'))
    assert sorted(res.keys()) == []

    res = feat('стали', morph.parse('стали'))

    assert 'GrammemePair:VERB,perf' in res
    assert 'GrammemePair:VERB,indc' in res
    assert 'GrammemePair:indc,perf' in res
    assert 'GrammemePair:NOUN,inan' in res
    assert 'GrammemePair:NOUN,femn' in res
    assert 'GrammemePair:NOUN,nomn' in res
    assert 'GrammemePair:gent,sing' in res
    assert 'GrammemePair:loct,sing' in res
    assert 'GrammemePair:femn,plur' in res
    assert 'GrammemePair:VERB,loct' not in res


def test_GrammemePair_threshold(morph):
    feat = features.GrammemePair(threshold=0.5)
    res = feat('стали', morph.parse('стали'))
    assert 'GrammemePair:VERB,plur' in res
    assert 'GrammemePair:NOUN,nomn' not in res

