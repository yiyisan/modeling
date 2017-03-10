#!/usr/bin/env python
# encoding: utf-8
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
from numpy.testing import assert_array_equal
import pytest
import work.marvin.dataPrepareforTraining.dataMapperPrepare as mapperBuilder
import work.marvin.dataPrepareforTraining.featureSelection as featureSelection
import joblib
import numpy as np
import pandas as pd
import sys
sys.path.append("../../")

def test_FtrTransFunc():
    f = mapperBuilder.FtrTransFunc('MinMaxScaler')
    assert type(f) is mapperBuilder.FtrTransFunc
    assert f is mapperBuilder.FtrTransFunc.MIN_MAX_SCALER
    with pytest.raises(ValueError):
        mapperBuilder.FtrTransFunc('NewTransform')

def test_dataMapperBuilder():
    mis_val = {"request_id": ("dropRow", "active")}
    traindf = pd.DataFrame({"request_id": [0, 1, 2],
                            "data": ["1", "0", "0"],
                            "Label": [1, 0, 0]})
    assert isinstance(traindf, pd.DataFrame)
    conti_ftr = traindf.describe().columns
    categ_ftr = [ "request_id"]
    dfm = mapperBuilder.dataMapperBuilder(traindf, categ_ftr, conti_ftr, mis_val=mis_val)
    ans = dfm.fit_transform(traindf[traindf.columns.difference(["Label"])], traindf["Label"])
    assert len(ans.columns) == 2


def test_contifeatureimportance():
    traindf = pd.DataFrame({"request_id": [0, 1, 2],
                            "data": [1, 0, 0],
                            "Label": [1, 0, 0]})
    featureImpr = featureSelection.continuousFeatureImpr(traindf, "Label")
    assert len(featureImpr) == 2
    print(featureImpr.ix[0, :].tolist())
    assert_array_equal(featureImpr.ix[0, :].tolist(), [np.inf, "data", 0.0, 0.866025403784])

