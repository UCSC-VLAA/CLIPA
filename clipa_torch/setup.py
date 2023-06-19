""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

def _read_reqs(relpath):
    fullpath = path.join(path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

REQUIREMENTS = _read_reqs("requirements.txt")

setup(
    name='clipa_torch',
    description='clipa pytorch implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/UCSC-VLAA/CLIPA',
    author='Zeyu Wang',
    author_email='zwang615@ucsc.edu',

    # Note that this is a string of words separated by whitespace, not a list.
    packages=find_packages(exclude=["figs*", "scripts*"]),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    python_requires='>=3.7',
)