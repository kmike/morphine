# -*- coding: utf-8 -*-
from __future__ import absolute_import

from six.moves import zip
from tabulate import tabulate
from tqdm import tqdm
import pycrfsuite

from morphine._fileresource import FileResource


class LessNoisyTrainer(pycrfsuite.Trainer):
    """
    This pycrfsuite.Trainer prints information about each iteration
    on a single line.
    """
    def on_iteration(self, log, info):
        if 'avg_precision' in info:
            print(("Iter {num:<3} "
                   "time={time:<5.2f} "
                   "loss={loss:<8.2f} "
                   "active={active_features:<5} "
                   "precision={avg_precision:0.3f}  "
                   "recall={avg_recall:0.3f}  "
                   "F1={avg_f1:0.3f}  "
                   "accuracy(item/instance)="
                   "{item_accuracy_float:0.3f} {instance_accuracy_float:0.3f}"
                ).format(**info).strip())
        else:
            print(("Iter {num:<3} "
                   "time={time:<5.2f} "
                   "loss={loss:<8.2f} "
                   "active={active_features:<5} "
                   "feature_norm={feature_norm:<8.2f} "
                ).format(**info).strip())


    def on_optimization_end(self, log):
        last_iter = self.logparser.last_iteration
        if 'scores' in last_iter:
            data = [
                [entity, score.precision, score.recall, score.f1, score.ref]
                for entity, score in sorted(last_iter['scores'].items())
            ]
            table = tabulate(data,
                headers=["Label", "Precision", "Recall", "F1", "Support"],
                # floatfmt="0.4f",
            )
            size = len(table.splitlines()[0])
            print("="*size)
            print(table)
            print("-"*size)
        super(LessNoisyTrainer, self).on_optimization_end(log)


class CRF(object):
    def __init__(self, algorithm=None, train_params=None, verbose=False,
                 model_filename=None, keep_tempfiles=False, trainer_cls=None):
        self.algorithm = algorithm
        self.train_params = train_params
        self.modelfile = FileResource(
            filename=model_filename,
            keep_tempfiles=keep_tempfiles,
            suffix=".crfsuite",
            prefix="model"
        )
        self.verbose = verbose
        self._tagger = None
        if trainer_cls is None:
            self.trainer_cls = pycrfsuite.Trainer
        else:
            self.trainer_cls = trainer_cls
        self.training_log_ = None

    def fit(self, X, y, X_dev=None, y_dev=None):
        """
        Train a model.

        Parameters
        ----------
        X : list of lists of dicts
            Feature dicts for several documents (in a python-crfsuite format).

        y : list of lists of strings
            Labels for several documents.

        X_dev : (optional) list of lists of dicts
            Feature dicts used for testing.

        y_dev : (optional) list of lists of strings
            Labels corresponding to X_dev.
        """
        if (X_dev is None and y_dev is not None) or (X_dev is not None and y_dev is None):
            raise ValueError("Pass both X_dev and y_dev to use the holdout data")

        if self._tagger is not None:
            self._tagger.close()
            self._tagger = None
        self.modelfile.refresh()

        trainer = self._get_trainer()
        train_data = zip(X, y)

        if self.verbose:
            train_data = tqdm(train_data, "loading training data to CRFsuite", len(X), leave=True)

        for xseq, yseq in train_data:
            trainer.append(xseq, yseq)

        if self.verbose:
            print("")

        if X_dev is not None:
            test_data = zip(X_dev, y_dev)

            if self.verbose:
                test_data = tqdm(test_data, "loading dev data to CRFsuite", len(X_dev), leave=True)

            for xseq, yseq in test_data:
                trainer.append(xseq, yseq, 1)

            if self.verbose:
                print("")

        trainer.train(self.modelfile.name, holdout=-1 if X_dev is None else 1)
        self.training_log_ = trainer.logparser
        return self

    def predict(self, X):
        """
        Make a prediction.

        Parameters
        ----------
        X : list of lists of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of lists of strings
            predicted labels

        """
        return list(map(self.predict_single, X))

    def predict_single(self, xseq):
        """
        Make a prediction.

        Parameters
        ----------
        xseq : list of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of strings
            predicted labels

        """
        return self.tagger.tag(xseq)

    def predict_marginals(self, X):
        """
        Make a prediction.

        Parameters
        ----------
        X : list of lists of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of lists of dicts
            predicted probabilities for each label at each position

        """
        return list(map(self.predict_marginals_single, X))

    def predict_marginals_single(self, xseq):
        """
        Make a prediction.

        Parameters
        ----------
        xseq : list of dicts
            feature dicts in python-crfsuite format

        Returns
        -------
        y : list of dicts
            predicted probabilities for each label at each position

        """
        labels = self.tagger.labels()
        self.tagger.set(xseq)
        return [
            {label: self.tagger.marginal(label, i) for label in labels}
            for i in range(len(xseq))
        ]

    @property
    def tagger(self):
        if self._tagger is None:
            if self.modelfile.name is None:
                raise Exception("Can't load model. Is the model trained?")

            tagger = pycrfsuite.Tagger()
            tagger.open(self.modelfile.name)
            self._tagger = tagger
        return self._tagger

    def _get_trainer(self):
        return self.trainer_cls(
            algorithm=self.algorithm,
            params=self.train_params,
            verbose=self.verbose,
        )

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct['_tagger'] = None
        return dct
