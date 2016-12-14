#!/usr/bin/env python
# encoding: utf-8
import pytest
import work.marvin.dataPrepareforTraining.dataMapperPrepare as mapperBuilder

def test_FtrTransFunc():
    f = mapperBuilder.FtrTransFunc('MinMaxScaler')
    assert type(f) is mapperBuilder.FtrTransFunc
    assert f is mapperBuilder.FtrTransFunc.MIN_MAX_SCALER
    with pytest.raises(ValueError):
        mapperBuilder.FtrTransFunc('NewTransform')
