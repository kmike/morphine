# -*- coding: utf-8 -*-
from __future__ import absolute_import


def sentence_start(sent, parsed_sent, i):
    if i == 0:
        yield 'BOS', 1.0


def sentence_end(sent, parsed_sent, i):
    if i == len(sent):
        yield 'EOS', 1.0


def token_identity(sent, parsed_sent, i):
    yield sent[i]






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
