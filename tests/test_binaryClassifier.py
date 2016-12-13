#!/usr/bin/env python
# encoding: utf-8
from work.marvin.binary_classifier_models.modelfit import BinaryClassifier
import work.marvin.binary_classifier_models
import work.marvin.binary_classifier_models.bestXgbModelProducer as bestXgbModelProducer


def testBinaryClassifier():
    xgb = BibaryClassifier("XGBOOST")
    assert xgb.model is bestXgbModelProducer
