import numpy as np

def side_of_line(px, py, ax, ay, bx, by, eps=1e-6):
    """
    Orientation test for point P vs. directed line A->B.
    Returns:
      1  : P is on the left side of A->B
     -1  : P is on the right side of A->B
      0  : P is collinear (within eps)
    """
    val = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(val) <= eps:
        return 0
    return 1 if val > 0 else -1
