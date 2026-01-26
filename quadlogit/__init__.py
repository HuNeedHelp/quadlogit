# -*- coding: utf-8 -*-
"""
quadlogit
===========

Version
-------
{version}
"""

__version__ = "0.2.1"

# Inject version into the module docstring shown by help(quadlogit)
__doc__ = (__doc__ or "").format(version=__version__)

from .api.quadlogit import fit
from . import demo
from .utils.helpers import generate_quad_indices
