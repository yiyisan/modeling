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
    assert combine(a) != None
    assert combine(b) != None
    assert combine(c) != None
