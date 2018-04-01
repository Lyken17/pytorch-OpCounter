#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages


readme = open('README.md').read()

VERSION = '0.0.22'

requirements = [
    'torch',
]

setup(
    # Metadata
    name='thop',
    version=VERSION,
    author='Ligeng Zhu',
    author_email='lykensyu+github@gmail.com',
    url='https://github.com/Lyken17/pytorch-OpCounter/',
    description='A tool to count the FLOPs of PyTorch model.',
    long_description=readme,
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)