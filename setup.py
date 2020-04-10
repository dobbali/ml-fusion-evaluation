#!/usr/bin/env python
# Learn more: https://github.com/pypa/sampleproject/blob/master/setup.py

from setuptools import setup, find_packages
from os import path


HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


about = {}
with open(path.join(HERE, 'fusion_evaluation', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), about)  # pylint: disable=exec-used

# If you have more than one package in the project, please see:
# https://github.com/pypa/sampleproject/blob/master/setup.py#L108-L117
PACKAGES = ['fusion_evaluation']

# Project Metadata
URL = 'https://github.rakops.com/americas-data/ml-fusion-evaluation'
AUTHOR = 'Manoj Dobbali'  # Add generic user name & email??
AUTHOR_EMAIL = 'manoj.dobbali@rakuten.com'

PYTHON_REQUIRES = '>=3.7'

INSTALL_REQUIRES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "datetime",
    "xgboost",
    "aws-secretsmanager-caching",
    "snowflake-sqlalchemy"
]


setup(
    # minimum requirements
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=long_description,
    packages=find_packages(exclude=['test*']),

    # ALL FIELDS BELOW ARE OPTIONAL #
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,

    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,

    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    # classifiers=[
    #     'Development Status :: 4 - Beta',
    #     'Intended Audience :: Information Technology',
    #     'Programming Language :: Python :: 3.6',
    #     'Topic :: Software Development :: Libraries',
    # ],
)
