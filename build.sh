#!/usr/bin/env bash
rm -rf build/ dist/
python setup.py bdist_wheel
twine check dist/*
twine upload dist/*