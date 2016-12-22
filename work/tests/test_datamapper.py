#!/usr/bin/env python
# encoding: utf-8
import pytest
import joblib
from sklearn_pandas import DataFrameMapper

def test_datamapepr():
    c_map = joblib.load("tests/fixtures/c_map")
    dfm = DataFrameMapper(c_map)
