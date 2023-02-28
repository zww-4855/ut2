"""
Unit and regression test for the UT2 package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import UT2


def test_UT2_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "UT2" in sys.modules
