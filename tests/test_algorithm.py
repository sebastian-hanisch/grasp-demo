"""grasp_algorithm.restarts: Budget-Buchführung, Monotonie der besten Tour, k=1-Wirkungslosigkeit der Neustarts
(jeder Neustart konstruiert + löst identisch), Streuung ab k>=2, Determinismus, Regressionsschutz."""

import numpy as np
import pytest

import grasp_algorithm as G
import grasp_tour as T


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return T.dist_matrix(xy)


def test_tour_stays_valid_and_length_matches_recomputed_length():
    D = _instance(20, 1)
    r = G.restarts(D, k=5, budget=20000, seed=3)
    assert sorted(r.best_tour.tolist()) == list(range(20))
    assert r.best_length == pytest.approx(T.tour_length(r.best_tour, D), abs=1e-6)


def test_budget_is_respected_and_the_first_restart_always_finishes():
    D = _instance(25, 2)
    budget = 3
    r = G.restarts(D, k=5, budget=budget, seed=1)                  # winziges Budget: der erste Abstieg muss trotzdem zu Ende laufen
    assert r.starts >= 1 and r.evaluations >= budget
    assert T.is_local_optimum(D, r.best_tour, "2opt")


def test_more_generous_budget_allows_more_restarts_and_never_worsens_the_best():
    D = _instance(25, 2)
    small = G.restarts(D, k=5, budget=20000, seed=1)
    large = G.restarts(D, k=5, budget=100000, seed=1)
    assert large.starts >= small.starts
    assert large.best_length <= small.best_length + 1e-6


def test_k_equals_one_makes_every_restart_identical_and_restarts_pointless():
    """Bei k=1 ist die Konstruktion deterministisch (immer dieselbe gierige Tour); nach demselben 2-opt-Abstieg
    ist deshalb JEDER Neustart identisch - die beste Tour nach Neustart 1 ist bereits die beste Tour am Ende."""
    D = _instance(20, 4)
    r = G.restarts(D, k=1, budget=100000, seed=5)
    assert r.starts >= 2                                            # genug Budget fuer mehrere (nutzlose) Neustarts
    assert len(set(round(x, 6) for x in r.restart_lengths)) == 1    # alle Neustarts liefern exakt dieselbe Laenge
    assert r.snapshots[0].tolist() == r.best_tour.tolist()          # der erste Neustart ist bereits das Endergebnis


def test_larger_k_produces_varying_restart_lengths():
    D = _instance(30, 6)
    r = G.restarts(D, k=8, budget=150000, seed=7)
    assert len(set(round(x, 6) for x in r.restart_lengths)) > 1


def test_deterministic_for_a_seed_and_different_for_another():
    D = _instance(25, 8)
    a = G.restarts(D, k=5, budget=30000, seed=11)
    b = G.restarts(D, k=5, budget=30000, seed=11)
    c = G.restarts(D, k=5, budget=30000, seed=40)
    assert np.array_equal(a.best_tour, b.best_tour) and a.evaluations == b.evaluations
    assert not np.array_equal(a.best_tour, c.best_tour) or a.evaluations != c.evaluations


def test_best_so_far_snapshots_are_monotone_non_increasing():
    D = _instance(25, 9)
    r = G.restarts(D, k=5, budget=60000, seed=2, keep_snapshots=True)
    lengths = [T.tour_length(s, D) for s in r.snapshots]
    assert all(b <= a + 1e-6 for a, b in zip(lengths, lengths[1:]))


def test_snapshots_are_only_kept_when_requested():
    D = _instance(15, 0)
    with_snaps = G.restarts(D, k=5, budget=20000, seed=0, keep_snapshots=True)
    without = G.restarts(D, k=5, budget=20000, seed=0, keep_snapshots=False)
    assert len(with_snaps.snapshots) == with_snaps.starts
    assert without.snapshots == []


def test_invalid_k_is_rejected():
    D = _instance(10, 0)
    with pytest.raises(ValueError):
        G.restarts(D, k=0, budget=1000, seed=0)
