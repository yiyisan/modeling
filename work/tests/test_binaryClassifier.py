#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.append("../../")
<<<<<<< HEAD

=======
>>>>>>> 608859a050f4c8a6c415d0e52feafda27cbe495e
from work.marvin.binary_classifier_models.modelfit import BinaryClassifier, HyperOpt
import work.marvin.binary_classifier_models
from sklearn.datasets import make_classification
from skopt.space import Categorical
from sklearn.ensemble import RandomForestClassifier

from work.marvin.dataPrepareforTraining.dataMapperPrepare import dataMapperBuilder


def testBinaryClassifier():
    xgb = BinaryClassifier("XGBOOST")
    assert xgb.model is work.marvin.binary_classifier_models.bestXgboostModelProducer
    rf = BinaryClassifier("RF")
    assert rf.model is work.marvin.binary_classifier_models.bestRfModelProducer
    lr = BinaryClassifier("LR")
    assert lr.model is work.marvin.binary_classifier_models.bestLrModelProducer
    gbm = BinaryClassifier("GBM")
    assert gbm.model is work.marvin.binary_classifier_models.bestGbdtModelProducer
    lgb = BinaryClassifier("LightGBM")
    assert lgb.model is work.marvin.binary_classifier_models.bestLightgbmModelProducer

def testOptimizeModel():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
    skopt_grid = {'max_features': (2, 19),
                  'min_samples_leaf': (50, 500),
                  'min_samples_split': (50, 500),
                  'n_estimators': (50, 800)}
<<<<<<< HEAD
    for mod in ["RF", "GBRT", "GP"]:       
        res = HyperOpt('RF').search(X, y, RandomForestClassifier, skopt_grid, 'neg_log_loss', n_calls=10)
        assert len(res) == 10


=======
    res = HyperOpt('RF').search(X, y, RandomForestClassifier, skopt_grid, 'neg_log_loss', n_calls=10)
>>>>>>> 608859a050f4c8a6c415d0e52feafda27cbe495e


