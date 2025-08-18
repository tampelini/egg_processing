import os
import sys
import types
import numpy as np

# Add project root to sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Provide a minimal stub for cv2 to satisfy calibrate import
sys.modules.setdefault('cv2', types.ModuleType('cv2'))

from app.calibrate import _solve_color_matrix


def test_solve_color_matrix_identity():
    measured = np.array([
        [10, 20, 30],
        [50, 80, 110],
        [200, 150, 100],
        [0, 255, 125],
    ], dtype=np.float32)
    target = measured.copy()
    M = _solve_color_matrix(measured, target)
    expected = np.hstack([np.eye(3), np.zeros((3, 1), dtype=np.float32)])
    np.testing.assert_allclose(M, expected, rtol=1e-5, atol=1e-5)
