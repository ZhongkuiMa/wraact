"""Test templates for activation function hull verification.

This module provides base test classes that can be reused across different
activation functions (ReLU, Sigmoid, Tanh, MaxPool, etc.).

Key Components:
===============
- BaseSoundnessTest: Reusable soundness verification tests
- BaseBasicTest: Reusable basic functionality tests
- BaseIntegrationTest: Reusable integration/pipeline tests

Usage Pattern:
==============
1. Import the base class
2. Create a test class inheriting from it
3. Implement required fixtures (activation_fn, hull_class_to_test)
4. All tests run automatically for your activation function

Example:
--------
from test_templates import BaseSoundnessTest
from wraact.acthull import LeakyReLUHull
import numpy as np

class TestLeakyReLUSoundness(BaseSoundnessTest):
    @pytest.fixture
    def activation_fn(self):
        def leakyrelu(x):
            return np.where(x >= 0, x, 0.01 * x)
        return leakyrelu

    @pytest.fixture
    def hull_class_to_test(self):
        return LeakyReLUHull
"""

__docformat__ = "restructuredtext"

from tests.test_templates.base_soundness_template import BaseSoundnessTest

__all__ = ["BaseSoundnessTest"]
