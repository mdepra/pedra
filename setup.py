#!/usr/bin/env python

from setuptools import setup
# from importlib_metadata import entry_points

# PACKAGE METADATA
##################
NAME = 'pedra'
FULLNAME = "PEDRA"
VERSION = '0.1'
DESCRIPTION = 'PipEline for Data Reduction of Asteroids'
with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())
AUTHOR = 'M. De Pra, M. Evangelista-Santana'
AUTHOR_EMAIL = 'mariondepra@gmail.com'
MAINTAINER = 'M. De Pra, M. Evangelista-Santana'
MAINTAINER_EMAIL = AUTHOR_EMAIL
URL = 'https://github.com/mdepra/pedra'
LICENSE = 'MIT License'

# TO BE INSTALLED
#################
PACKAGES = ['pedra']

#PACKAGE_DATA = {'pedra.data':['jwst/*']}

# SCRIPTS = ['scripts/pedragui']

# DEPENDENCIES
##############
INSTALL_REQUIRES = [
    # 'cana-asteroids',
    'pandas',
    'scipy',
    'numpy',
    'matplotlib',
    'pyyaml',
    'astropy',
    'photutils',
    'rebin',
    'twirl',
    'reproject',
    'ipywidgets',
    'ipympl',
    'customtkinter'
]

PYTHON_REQUIRES = ">=3.6"


# INSTALL SCRIPT
################
if __name__ == '__main__':
    setup(name=NAME,
          description=DESCRIPTION,
          # long_description=LONG_DESCRIPTION,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          license=LICENSE,
          url=URL,
          # platforms=PLATFORMS,
          # scripts=SCRIPTS,
          packages=PACKAGES,
          # dependency_links = DEPENDENCY_LINKS,
          # entry_points = {
          #                 'console_scripts' : [
          #                                      'pedragui = pedra.script:main',
          #                                     ]
          #                },
          # ext_modules=EXT_MODULES,
          # package_data=PACKAGE_DATA,
          # classifiers=CLASSIFIERS,
          # keywords=KEYWORDS,
          # cmdclass=CMDCLASS,
          install_requires=INSTALL_REQUIRES,
          python_requires=PYTHON_REQUIRES,
          )
