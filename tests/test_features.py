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
    assert sorted(res.keys()) == ['Grammeme']
    assert sorted(res['Grammeme'].keys()) == ['INTJ', 'PRCL', 'PREP']
    assert res['Grammeme']['PREP'] > res['Grammeme']['PRCL']
    assert res['Grammeme']['PREP'] > res['Grammeme']['INTJ']

    res = feat('стали', morph.parse('стали'))
    assert 'past' in res['Grammeme']
    assert 'accs' in res['Grammeme']
    assert res['Grammeme']['VERB'] > res['Grammeme']['NOUN']


def test_Grammeme_threshold(morph):
    feat = features.Grammeme(threshold=0.1)
    res = feat('на', morph.parse('на'))
    assert sorted(res['Grammeme'].keys()) == ['PREP']
    assert res['Grammeme']['PREP'] > 0.99


def test_GrammemePair(morph):
    feat = features.GrammemePair(threshold=0)
    res = feat('на', morph.parse('на'))
    assert sorted(res.keys()) == ['GrammemePair']
    assert res['GrammemePair'] == {}

    res = feat('стали', morph.parse('стали'))
    assert sorted(res.keys()) == ['GrammemePair']

    res_grp = res['GrammemePair']

    assert 'VERB,perf' in res_grp
    assert 'VERB,indc' in res_grp
    assert 'indc,perf' in res_grp
    assert 'NOUN,inan' in res_grp
    assert 'NOUN,femn' in res_grp
    assert 'NOUN,nomn' in res_grp
    assert 'gent,sing' in res_grp
    assert 'loct,sing' in res_grp
    assert 'femn,plur' in res_grp
    assert 'VERB,loct' not in res_grp


def test_GrammemePair_threshold(morph):
    feat = features.GrammemePair(threshold=0.5)
    res = feat('стали', morph.parse('стали'))
    assert 'VERB,plur' in res['GrammemePair']
    assert 'NOUN,nomn' not in res['GrammemePair']

