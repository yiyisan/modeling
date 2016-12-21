#!/usr/bin/env python
# encoding: utf-8

import pytest
import joblib
from skopt import gp_minimize
from work.marvin.binary_classifier_models.modelfit import skopt_search
from sklearn.ensemble import RandomForestClassifier
import os


def test_opt():
    print(os.getcwd())
    X_train = joblib.load("tests/X_train")
    y_train = joblib.load("tests/y_train")
    param_grid = joblib.load("tests/param_grid")
    search_func_args = joblib.load("tests/search_func_args")
    results = skopt_search('GP').search(
    X_train,
    y_train,
    RandomForestClassifier,
    param_grid,
    'neg_log_loss',
    **search_func_args)

