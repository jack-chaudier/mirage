from __future__ import annotations

import numpy as np


def _same_context(a, b) -> bool:
    if a.d_total != b.d_total:
        return False
    a_inf = np.isneginf(a.W)
    b_inf = np.isneginf(b.W)
    if not np.array_equal(a_inf, b_inf):
        return False
    mask = ~a_inf
    return bool(np.allclose(a.W[mask], b.W[mask], atol=1e-12, rtol=0.0))
