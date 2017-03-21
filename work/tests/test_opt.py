#!/usr/bin/env python
# encoding: utf-8

import pytest
import joblib
from skopt import gp_minimize
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from work.marvin.binary_classifier_models.modelfit import HyperOpt
from work.marvin.binary_classifier_models.bestLightgbmModelProducer import configSpaceInitialization as lgbconfigspace
from work.marvin.binary_classifier_models.bestGbdtModelProducer import configSpaceInitialization as gbmconfigspace
from work.marvin.binary_classifier_models.bestXgboostModelProducer import configSpaceInitialization as xgbconfigspace
from work.marvin.binary_classifier_models.bestRfModelProducer import configSpaceInitialization as rfconfigspace
import os


def test_hyperopt():
    X_train, y_train = make_classification(random_state=27)
    param_grid = lgbconfigspace(X_train.shape)
    assert param_grid is not None
    param_grid = gbmconfigspace(X_train.shape)
    assert param_grid is not None
    param_grid = xgbconfigspace(X_train.shape)
    assert param_grid is not None
    param_grid = rfconfigspace(X_train.shape)
    assert param_grid is not None
    results = HyperOpt('GP').search(
        X_train,
        y_train,
        RandomForestClassifier,
        param_grid,
		n_calls=10)
    assert results is not None
