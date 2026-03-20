"""Shared fixtures for VISTA tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def simple_dataset_2d():
    """A minimal 2-alternative, 2-criteria dataset."""
    return np.array([[0.5, 0.5], [0.7, 0.8]])


@pytest.fixture
def reference_point_2d():
    return np.array([0.5, 0.5])
