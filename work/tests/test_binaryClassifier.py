#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.append("../../")
import pandas as pd
import numpy as np
import work.marvin.binary_classifier_models
from work.marvin.binary_classifier_models.modelfit import BinaryClassifier, HyperOpt
from sklearn.datasets import make_classification
from skopt.space import Categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn2pmml.decoration import ContinuousDomain
from sklearn.preprocessing import Imputer
from sklearn_pandas import DataFrameMapper

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

def testHyperOpt():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
    skopt_grid = {'max_features': (2, 19),
                  'min_samples_leaf': (50, 500),
                  'min_samples_split': (50, 500),
                  'n_estimators': (50, 800)}
    for mod in ["RF", "GBRT", "GP"]:
        res = HyperOpt('RF').search(X, y, RandomForestClassifier, skopt_grid, 'neg_log_loss', n_calls=10)
        assert len(res) == 19

def testOptimizeBestModel():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
    Xall = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=['target'])], axis=1)
    conti_ftr = list(range(20))
    datamapper = DataFrameMapper([(conti_ftr, [ContinuousDomain(invalid_value_treatment='as_is',
                                                     missing_value_treatment='as_mean'),
                                   Imputer()])], df_out=True)
    X_ = datamapper.fit_transform(X)
    lgb = BinaryClassifier("LightGBM")
    bestskopt, trace = lgb.optimizeBestModel(Xall, datamapper=datamapper, target='target', search_alg="GP", n_calls=10)
