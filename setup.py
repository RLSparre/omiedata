# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='omiedata',
    version='0.1.0',
    description='Package to read data from https://www.omie.es/',
    long_description=readme,
    author='Rasmus Lodberg Sparre',
    author_email='rasmuslodbergsparre@gmail.com',
    url='https://github.com/RLSparre/omiedata',
    license=license,
    packages=find_packages(exclude=('tests')),
    include_package_data=True
)