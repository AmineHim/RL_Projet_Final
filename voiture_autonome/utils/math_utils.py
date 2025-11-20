# utils/math_utils.py

import numpy as np


def euclidean_distance(a, b):
    """
    Retourne la distance euclidienne entre deux points 3D (comme carla.Location ou numpy array).
    """
    a = np.array([a.x, a.y, a.z]) if hasattr(a, "x") else np.array(a)
    b = np.array([b.x, b.y, b.z]) if hasattr(b, "x") else np.array(b)
    return np.linalg.norm(a - b)


def angle_between(v1, v2):
    """
    Calcule l'angle entre deux vecteurs 2D ou 3D.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return np.arccos(np.clip(dot / norm, -1.0, 1.0))
