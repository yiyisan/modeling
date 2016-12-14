#!/usr/bin/env python
# encoding: utf-8
from work.marvin.binary_classifier_models.modelfit import BinaryClassifier
import work.marvin.binary_classifier_models


def testBinaryClassifier():
    xgb = BinaryClassifier("XGBOOST")
    assert xgb.model is work.marvin.binary_classifier_models.bestXgboostModelProducer
    rf = BinaryClassifier("RF")
    assert rf.model is work.marvin.binary_classifier_models.bestRfModelProducer
