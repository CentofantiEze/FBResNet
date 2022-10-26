#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import os

__name__ = 'FBRN'

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        __name__, 'info.py'))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)

setup(
      name=__name__,
      version=release_info['__version__'],
      description='Forward Backwards Residual Neural Network',
      url=release_info['__url__'],
      author=release_info['__author__'],
      author_email=release_info['__email__'],
      #license=release_info['__license__'],
      packages=['FBRN'],
      zip_safe=False,
)
