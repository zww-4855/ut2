"""
Unit and regression test for the UT2 package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import UT2

import numpy as np
import pyscf
from pyscf import ao2mo

from UT2.run_ccd import * 

def test_UT2_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "UT2" in sys.modules
    assert "pyscf" in sys.modules

