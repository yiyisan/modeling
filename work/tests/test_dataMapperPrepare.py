#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.append("../../")
import pytest
import work.marvin.dataPrepareforTraining.dataMapperPrepare as mapperBuilder
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
import joblib
import pandas as pd

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
    print(len(conti_ftr))
    print(len(categ_ftr), categ_ftr)
    dfm = mapperBuilder.dataMapperBuilder(traindf, categ_ftr, conti_ftr, mis_val=mis_val)
    print(traindf.columns.difference(["Label"]))
    ans = dfm.fit_transform(traindf[traindf.columns.difference(["Label"])], traindf["Label"])
    assert len(ans.columns) == 2


def test_naming_hump_to_underline():
    samples = [["asMising", "as_mising"], ["ming", "ming"], ["thereAreTwo", "there_are_two"]]
    for ls in samples:
        assert mapperBuilder.naming_hump_to_underline(ls[0]) == ls[1]
