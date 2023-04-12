import setuptools
from setuptools import setup

setup(
    name='event',
    version='1.0',
    description='LM reasoning',
    packages=setuptools.find_packages(),
    install_requires=['torch==1.13.1','transformers==4.25.1','datasets','matplotlib', 'more_itertools', 'sentencepiece', 'protobuf']
    )