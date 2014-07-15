# -*- coding: utf-8 -*-
from __future__ import absolute_import
import inspect


def func_takes_argument(func, argname):
    """
    Return True if function ``func`` takes keyword argument named ``argname``:

    >>> def foo(x, y): pass
    >>> def bar(x, y, z): pass
    >>> def baz(x, **kwargs): pass
    >>> func_takes_argument(foo, 'x')
    True
    >>> func_takes_argument(foo, 'y')
    True
    >>> func_takes_argument(foo, 'z')
    False
    >>> func_takes_argument(bar, 'z')
    True
    >>> func_takes_argument(baz, 'z')
    True
    """
    spec = inspect.getargspec(func)
    return spec.keywords is not None or argname in spec.args
