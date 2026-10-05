"""Orakel-Tests mit anderem Rechenweg: jeder Nachbar (2-opt, Tausch, Or-opt) per Neuaufbau der Tour + volle Längenberechnung statt Delta-Matrix,
Zahl der Nachbarn, Kreuzungszählung per Segmentschnitt, 1-Baum per networkx-Spannbaum, Held-Karp-Schranke gegen exakte Bitmasken-DP, die
RCL-Konstruktion als unabhängiger Nachbau (mit festgelegten Zufallsentscheidungen) und die Buchführung eines GRASP-Laufs."""
import itertools

import numpy as np
import pytest

import grasp_algorithm as G
import grasp_construct as CO
import grasp_constants as C
import grasp_tour as T


def _inst(n, seed):
    xy = np.random.default_rng(seed).random((n, 2)) * 100
    return T.dist_matrix(xy), xy


def _len(t, D):
    t = [int(x) for x in t]
    return sum(D[t[i], t[(i + 1) % len(t)]] for i in range(len(t)))


def _neighbors(t, D, kind):
    """Längenänderungen aller Nachbarn der Art `kind`, jede per Neuaufbau der Tour und voller Längenberechnung."""
    t = [int(x) for x in t]
    n, base, out = len(t), _len(t, D), []
    if kind in ("2opt", "swap"):
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                new = t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:] if kind == "2opt" else [t[j] if p == i else t[i] if p == j else t[p] for p in range(n)]
                out.append(_len(new, D) - base)
        return out
    L = int(kind[-1])
    for i in range(n):
        seg = [t[(i + k) % n] for k in range(L)]
        rest = [t[(i + L + k) % n] for k in range(n - L)]
        for m in range(n):
            if ((m - (i - 1)) % n) < L + 1:
                continue
            q = rest.index(t[m])
            out.append(min(_len(rest[:q + 1] + s + rest[q + 1:], D) - base for s in (seg, seg[::-1])))
    return out


KINDS = {"swap": ("swap",), "2opt": ("2opt",), "oropt": ("oropt1", "oropt2", "oropt3"), "2opt+oropt": ("2opt", "oropt1", "oropt2", "oropt3")}


@pytest.mark.parametrize("n,seed", [(5, 1), (6, 2), (7, 3), (9, 4), (12, 5), (14, 6)])
def test_every_neighbor_delta_and_the_best_move_match_full_re_evaluation(n, seed):
    D, _ = _inst(n, seed)
    t = T.random_tour(n, np.random.default_rng(seed))
    for nbh, kinds in KINDS.items():
        kinds = [k for k in kinds if not k.startswith("oropt") or n >= int(k[-1]) + 3]
        for kind in kinds:
            nb = "oropt" if kind.startswith("oropt") else kind
            (_, delta, ok, _), = [c for c in T._candidates(t, D, nb) if c[0] == kind]
            assert sorted(delta[ok].tolist()) == pytest.approx(sorted(_neighbors(t, D, kind)), abs=1e-9)   # gleiche Nachbarzahl, gleiche Deltas
        allc = [d for k in kinds for d in _neighbors(t, D, k)]
        move, evals = T.find_move(t, D, nbh, "best")
        assert (move[-1] if move else 0.0) == pytest.approx(min(allc + [0.0]), abs=1e-9)
        if move is None:
            assert evals == len(allc)                                                       # Zähler = Zahl der Nachbarn
        else:
            new = T.apply_move(t, move)
            assert sorted(new.tolist()) == list(range(n)) and _len(new, D) - _len(t, D) == pytest.approx(move[-1], abs=1e-9)


@pytest.mark.parametrize("nbh,rule", [("2opt", "first"), ("oropt", "best"), ("2opt+oropt", "first"), ("swap", "best")])
def test_descent_ends_in_a_true_local_optimum(nbh, rule):
    D, _ = _inst(14, 7)
    r = T.descend(D, T.random_tour(14, np.random.default_rng(1)), nbh, rule)
    assert r.length == pytest.approx(_len(r.tour, D), abs=1e-9)
    assert min([d for k in KINDS[nbh] for d in _neighbors(r.tour, D, k)] + [0.0]) >= -1e-9


def _intersect(a, b, c, d):
    def o(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return o(c, d, a) * o(c, d, b) < 0 and o(a, b, c) * o(a, b, d) < 0


@pytest.mark.parametrize("seed", range(6))
def test_crossing_count_matches_pairwise_segment_intersection(seed):
    D, xy = _inst(18, seed)
    t = T.random_tour(18, np.random.default_rng(seed)).tolist()
    cnt = sum(_intersect(xy[t[i]], xy[t[(i + 1) % 18]], xy[t[j]], xy[t[(j + 1) % 18]]) for i in range(18) for j in range(i + 1, 18)
              if len({t[i], t[(i + 1) % 18], t[j], t[(j + 1) % 18]}) == 4)
    assert T.count_crossings(xy, t) == cnt


def test_one_tree_and_held_karp_bound_against_networkx_and_exact_dp():
    nx = pytest.importorskip("networkx")
    for seed in range(6):
        n = 9
        D, _ = _inst(n, seed)
        pi = np.random.default_rng(seed).normal(0, 5, n)
        Cm = D + pi[:, None] + pi[None, :]
        cost, deg = T._one_tree(Cm)
        g = nx.Graph()
        g.add_weighted_edges_from((i, j, Cm[i, j]) for i in range(1, n) for j in range(i + 1, n))
        assert cost == pytest.approx(nx.minimum_spanning_tree(g).size(weight="weight") + sum(sorted(Cm[0, 1:])[:2]), abs=1e-9)
        opt = min(sum(D[a, b] for a, b in zip((0,) + p, p + (0,))) for p in itertools.permutations(range(1, n)))
        ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
        assert T.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS) <= opt + 1e-7           # gültige untere Schranke
        assert G.restarts(D, 3, 5000, seed).best_length >= opt - 1e-9


class _FixedRng:
    def __init__(self, picks):
        self.picks, self.i = picks, 0

    def integers(self, m):
        v = self.picks[self.i % len(self.picks)] % m
        self.i += 1
        return v


@pytest.mark.parametrize("n,k,seed", [(6, 1, 1), (9, 2, 2), (12, 3, 3), (15, 5, 4), (8, 20, 5), (20, 3, 6)])
def test_rcl_construction_matches_an_independent_rebuild(n, k, seed):
    D, _ = _inst(n, seed)
    picks = np.random.default_rng(seed).integers(0, 50, n).tolist()
    tour, vis, cur = [0], {0}, 0
    for step in range(n - 1):
        rcl = sorted((j for j in range(n) if j not in vis), key=lambda j: (D[cur, j], j))[:k]
        nxt = rcl[picks[step % len(picks)] % len(rcl)]
        tour.append(nxt)
        vis.add(nxt)
        cur = nxt
    assert CO.construct(D, k, _FixedRng(picks)).tolist() == tour


@pytest.mark.parametrize("n,k,budget,seed", [(12, 1, 800, 0), (20, 2, 4000, 1), (30, 3, 20000, 2), (15, 5, 600, 3)])
def test_restart_bookkeeping_holds_up_under_recomputation(n, k, budget, seed):
    D, _ = _inst(n, seed)
    run = G.restarts(D, k, budget, seed, keep_snapshots=True, trace_points=40)
    assert sorted(run.best_tour.tolist()) == list(range(n)) and run.best_length == pytest.approx(_len(run.best_tour, D), abs=1e-9)
    assert run.best_length == pytest.approx(min(run.restart_lengths), abs=1e-9)
    sl = [_len(s, D) for s in run.snapshots]
    assert all(a >= b - 1e-9 for a, b in zip(sl, sl[1:])) and sl[-1] == pytest.approx(run.best_length, abs=1e-9)
    assert run.starts == len(run.restart_lengths) == len(run.snapshots) and run.trace_iter[-1] == run.evaluations
