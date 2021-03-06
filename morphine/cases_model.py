# -*- coding: utf-8 -*-
from __future__ import absolute_import

from pymorphy2 import MorphAnalyzer
from pymorphy2.tagset import OpencorporaTag

from morphine import features
from morphine.feature_extractor import FeatureExtractor
from morphine.basetagger import PartialTagger


class CaseFeatureExtractor(FeatureExtractor):
    IGNORE = {'anim', 'inan', 'masc', 'femn', 'neut', 'intg', 'real', 'Ms-f'}

    def __init__(self):
        super(CaseFeatureExtractor, self).__init__(
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
                features.Pattern([-2, 'token_lower']),

                features.Pattern([-1, 'Grammeme']),
                features.Pattern([+1, 'Grammeme']),

                features.Pattern([-2, 'Grammeme'], [-1, 'Grammeme']),
                features.Pattern([-1, 'Grammeme'], [0, 'Grammeme']),
                features.Pattern([-1, 'Grammeme'], [0, 'GrammemePair']),

                features.Pattern([-1, 'GrammemePair']),
                features.Pattern([+1, 'GrammemePair']),
            ],
        )


class Tagger(PartialTagger):
    def outval(self, tag):
        case = str(tag.case or 'NA')
        return OpencorporaTag.RARE_CASES.get(case, case)


class MultiTagger(PartialTagger):
    def outval(self, tag):
        case = str(tag.case or '')
        case = OpencorporaTag.RARE_CASES.get(case, case)

        num = str(tag.number or '')
        tran = str(tag.transitivity or '')

        return '-'.join([tag._POS, case, num, tran])


# # @six.add_metaclass(abc.ABCMeta)
# class Tagger(object):
#     def __init__(self, fe, crf):
#         self.fe = fe
#         self.crf = crf
#
#     def outval(self, tag):
#         case = str(tag.case or 'NA')
#         return OpencorporaTag.RARE_CASES.get(case, case)
#
#     # @accepts_untokenized
#     def predict(self, tokens):
#         if not isinstance(tokens, (list, tuple)):
#             tokens = simple_word_tokenize(tokens)
#
#         parsed_tokens, xseq = self.fe.transform_and_parse_single(tokens)
#         yseq = self.crf.tagger.tag(xseq)
#         res = []
#
#         for i, (parses, case) in enumerate(zip(parsed_tokens, yseq)):
#             p = max(parses, key=lambda p: self.outval(p.tag) == case)
#             if case != 'NA':
#                 p = p._replace(score=self.crf.tagger.marginal(case, i))
#             res.append(p)
#
#         return res

    # def predict_proba(self, tokens):
    #         # if case != 'NA':
    #         #     p = p._replace(score=self.crf.tagger.marginal(case, i))
    #
    #         # res.append(parses_sorted)
    #
    #     cases = self.tagger.tag(self.feature_extractor.transform_sent(tokens))
    #
    #     i = 0
    #     res = []
    #     for parses in pymorphy_res:
    #         if self.strip_pnct and 'PNCT' in parses[0].tag:
    #             res.append(parses[0])
    #             continue
    #
    #         case = cases[i]
    #
    #         def keyfunc(p):
    #             if case == 'NA':
    #                 match = p.tag.case is None
    #             else:
    #                 match = case in self.morph.TagClass.fix_rare_cases(p.tag.grammemes)
    #
    #             return match, p.score
    #
    #         p = max(parses, key=keyfunc)
    #         if case != 'NA':
    #             p = p._replace(score=self.tagger.marginal(case, i))
    #         res.append(p)
    #         i += 1
    #
    #     return res



# class Tagger(object):
#
#     def __init__(self, model_filename, morph=None, feature_extractor=None,
#                  model_strips_pnct=True):
#         if morph is None:
#             morph = MorphAnalyzer()
#         self.morph = morph
#
#         self.tagger = pycrfsuite.Tagger()
#         self.tagger.open(model_filename)
#
#         if feature_extractor is None:
#             feature_extractor = CaseFeatureExtractor(morph)
#         self.feature_extractor = feature_extractor
#
#         self.strip_pnct = model_strips_pnct
#
#     def predict(self, tokens):
#         if not isinstance(tokens, (list, tuple)):
#             tokens = simple_word_tokenize(tokens)
#
#         pymorphy_res = [self.morph.parse(tok) for tok in tokens]
#
#         if self.strip_pnct:
#             tokens = [token for token, parses in zip(tokens, pymorphy_res)
#                       if 'PNCT' not in parses[0].tag]
#
#         cases = self.tagger.tag(self.feature_extractor.transform_sent(tokens))
#
#         i = 0
#         res = []
#         for parses in pymorphy_res:
#             if self.strip_pnct and 'PNCT' in parses[0].tag:
#                 res.append(parses[0])
#                 continue
#
#             case = cases[i]
#
#             def keyfunc(p):
#                 if case == 'NA':
#                     match = p.tag.case is None
#                 else:
#                     match = case in self.morph.TagClass.fix_rare_cases(p.tag.grammemes)
#
#                 return match, p.score
#
#             p = max(parses, key=keyfunc)
#             if case != 'NA':
#                 p = p._replace(score=self.tagger.marginal(case, i))
#             res.append(p)
#             i += 1
#
#         return res
