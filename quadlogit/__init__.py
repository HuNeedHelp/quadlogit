# -*- coding: utf-8 -*-
"""
quadlogit
===========

quadlogit is a Python package for .......?

The package accompanies the paper: ??

Quick links
-----------
- Documentation: https://quadlogit.readthedocs.io/
- Source code & replication materials (GitHub): https://github.com/zizhongyan/quadlogit

Main entry point
----------------
??


Notes
-----
??


Version
-------
{version}
"""

__version__ = "0.2.1"

# Inject version into the module docstring shown by help(quadlogit)
__doc__ = (__doc__ or "").format(version=__version__)

from .api.quadlogit import fit
from . import demo
from .utils.helpers import loadindex


