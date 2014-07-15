#!/bin/sh
py.test --cov morphine --cov-report html --cov-report term --doctest-modules --ignore setup.py "$@"
