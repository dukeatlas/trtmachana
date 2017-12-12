#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os.path

from setuptools import find_packages
from setuptools import setup

def get_version():
    g = {}
    exec(open(os.path.join("trtmachana", "version.py")).read(), g)
    return g["__version__"]

setup(name = "trtmachana",
      version = get_version(),
      packages = find_packages(exclude = ["tests"]),
      scripts = [],
      data_files = [os.path.join("trtmachana", "description.txt"), os.path.join("trtmachana", "long_description.txt")],
      description = open(os.path.join("trtmachana", "description.txt")).read().strip(),
      long_description = open(os.path.join("trtmachana", "long_description.txt")).read().strip(),
      author = "Doug Davis",
      author_email = "ddavis@cern.ch",
      maintainer = "Doug Davis",
      maintainer_email = "ddavis@cern.ch",
      url = "https://github.com/dukeatlas/trtmachana",
      download_url = "https://github.com/dukeatlas/trtmachana/releases",
      license = "MIT",
      test_suite = "tests",
      install_requires = ["uproot>=2.5.10","matplotlib","scikit-learn","pandas"],
      tests_require = ["uproot>=2.5.10","matplotlib","scikit-learn","pandas"],
      classifiers = [
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Physics",
      ],
      platforms = "Any",
)
