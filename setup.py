#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup(
    author="Dave Pacia",
    author_email='davepacia15@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Exploration of the Abalone dataset using Scikit-Learn library.",
    entry_points={
        'console_scripts': [
            'abalone_dpacia_sklearn=abalone_dpacia_sklearn.cli:main',
        ],
    },
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='abalone_dpacia_sklearn',
    name='abalone_dpacia_sklearn',
    packages=find_packages(include=['abalone_dpacia_sklearn']),
    test_suite='tests',
    url='https://github.com/dpacia/abalone_dpacia_sklearn',
    version='0.1.0',
    zip_safe=False,
)
