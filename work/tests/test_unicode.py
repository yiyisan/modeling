# coding: utf-8
import six
import pytest

def combine(value):
    if isinstance(value, six.string_types):
        u"{}_{}".format(1, value)
    else:
        "{}_{}".format(1, value)


def test_combine():
    a = u'你'
    b = 'adsad'
    c = 1
    assert combine(a) is not None
    assert combine(b) is not None
    assert combine(c) is not None
