#!/usr/bin/env python
# encoding: utf-8

import pytest
import work.marvin.binary_classification_evaluation.binaryClassEvaluationPlots as evaluationPlots


def test_kscurve():
    labels_test = [1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                   1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.]
    test_predprob = [0.22297297, 0.22297297, 0.22297297, 0.22297297, 0.22297297, 0.22297297,
                     0.22297297, 0.22297297, 0.22297297, 0.22297297, 0.22297297, 0.22297297,
                     0.22297297, 0.22297297, 0.22297297, 0.22297297, 0.22297297, 0.22297297,
                     0.22297297, 0.22297297, 0.22297297, 0.22297297, 0.22297297, 0.22297297,
                     0.22297297]
    with pytest.raises(IndexError):
        iks_val, ks_x, p, r = evaluationPlots.ks_curve(labels_test, test_predprob, "tests/ks.png")
