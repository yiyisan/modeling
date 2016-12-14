# coding: utf-8
import six
import pytest

def combine(value):
    if isinstance(value, six.string_types):
        u"{}={}".format(1, value)
    else:
        "{}={}".format(1, value)


def test_combine():
    a = u'你'
    b = 'adsad'
    c = 1
    combine(a)
    combine(b)
    combine(c)

if __name__ == "__main__":
    test_combine()
