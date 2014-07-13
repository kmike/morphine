#!/usr/bin/env python
from setuptools import setup

setup(
    name = 'morphine',
    version = '0.0',
    author = 'Mikhail Korobov',
    author_email = 'kmike84@gmail.com',
    url = 'https://github.com/kmike/morphine/',
    description = 'Disambiguation engine for pymorphy2',
    long_description = open('README.rst').read(),
    license = 'MIT license',
    packages = ['morphine'],
    requires=['pymorphy2', 'pycrfsuite', 'toolz'],
    install_requires=[
        'python-crfsuite >= 0.6.1',
        'toolz >= 0.7',
        'pymorphy2 >= 0.8',
    ],
    extras_require = {
        'fast':  [
            "DAWG >= 0.7.3",
            "cytoolz >= 0.7",
        ],
    },
    classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: Russian',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Text Processing :: Linguistic',
    ],
)
