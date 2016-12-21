#!/usr/bin/env python
# encoding: utf-8
from work.marvin.binary_classifier_models.modelfit import BinaryClassifier, skopt_search
import work.marvin.binary_classifier_models
from sklearn.datasets import make_classification
from skopt.space import Categorical
from sklearn.ensemble import RandomForestClassifier


def testBinaryClassifier():
    xgb = BinaryClassifier("XGBOOST")
    assert xgb.model is work.marvin.binary_classifier_models.bestXgboostModelProducer
    rf = BinaryClassifier("RF")
    assert rf.model is work.marvin.binary_classifier_models.bestRfModelProducer
    lr = BinaryClassifier("LR")
    assert lr.model is work.marvin.binary_classifier_models.bestLrModelProducer
    gbm = BinaryClassifier("GBM")
    assert gbm.model is work.marvin.binary_classifier_models.bestGbdtModelProducer

def testOptimizeModel():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
    skopt_grid = {'max_features': (2, 19),
                  'min_samples_leaf': (50, 500),
                  'min_samples_split': (50, 500),
                  'n_estimators': (50, 800)}
    res = skopt_search('RF').search(X, y, RandomForestClassifier, skopt_grid, 'neg_log_loss', n_calls=10)
