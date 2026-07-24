"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest
import numpy as np

from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.robots.three_link_planar import create_three_link_planar


@pytest.fixture
def two_link_robot():
    """Create a standard 2-link planar robot with equal link lengths."""
    return create_two_link_planar(link1=1.0, link2=1.0)


@pytest.fixture
def two_link_unequal():
    """Create a 2-link planar robot with unequal link lengths."""
    return create_two_link_planar(link1=1.0, link2=0.8)


@pytest.fixture
def three_link_robot():
    """Create a standard 3-link planar robot."""
    return create_three_link_planar(link1=1.0, link2=1.0, link3=0.5)
