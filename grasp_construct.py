"""Die randomisierte gierige Konstruktion von GRASP (Feo & Resende 1989/1995): direkte Verallgemeinerung von
`grasp_tour.nearest_neighbor_tour` (das an jedem Schritt IMMER den nächsten unbesuchten Knoten wählt). Hier werden
stattdessen die `k` nächsten unbesuchten Knoten als Restricted Candidate List (RCL) gebildet, und einer davon
zufällig gewählt - das macht jede Konstruktion anders, statt bei jedem Neustart dieselbe (rein gierige) Tour zu
liefern. Bei `k=1` ist die RCL immer genau ein Knoten (der nächste): das Ergebnis ist dann exakt
`nearest_neighbor_tour(D)` - deterministisch, kein Zufall (siehe tests/test_construct.py)."""

import numpy as np


def construct(D, k, rng):
    """Eine randomisierte gierige Tour, beginnend am Depot (Knoten 0, wie alle Touren dieser Linie). `k` = Größe
    der Restricted Candidate List je Schritt (bei weniger als `k` übrigen Knoten: alle übrigen)."""
    if k < 1:
        raise ValueError(k)
    n = len(D)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    cur = 0
    for _ in range(n - 1):
        remaining = np.where(~visited)[0]
        dists = D[cur, remaining]
        order = np.argsort(dists, kind="stable")
        rcl = remaining[order[:min(k, len(remaining))]]
        nxt = int(rcl[rng.integers(len(rcl))])
        tour.append(nxt)
        visited[nxt] = True
        cur = nxt
    return np.array(tour, dtype=np.int64)
