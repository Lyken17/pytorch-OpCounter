#!/usr/bin/env python
import os, sys
import shutil
import datetime

from setuptools import setup, find_packages
from setuptools.command.install import install

readme = open('README.md').read()

VERSION = '0.0.31'

requirements = [
    'torch',
]

# import subprocess
# commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True).decode('UTF-8').rstrip()
# VERSION += "_" + str(int(commit_hash, 16))[:8]
VERSION += "_" + datetime.datetime.now().strftime('%Y%m%d%H%M')[2:]
# print(VERSION)

setup(
    # Metadata
    name='thop',
    version=VERSION,
    author='Ligeng Zhu',
    author_email='lykensyu+github@gmail.com',
    url='https://github.com/Lyken17/pytorch-OpCounter/',
    description='A tool to count the FLOPs of PyTorch model.',
    long_description=readme,
    long_description_content_type='text/markdown',
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
