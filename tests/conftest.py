# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pytest
import pymorphy2


@pytest.fixture(scope='session')
def morph():
    return pymorphy2.MorphAnalyzer()
