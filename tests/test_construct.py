"""grasp_construct.construct: gültige Permutation, bei k=1 exakte Übereinstimmung mit grasp_tour.nearest_neighbor_tour
(Regressionstest gegen die bereits geprüfte Funktion, keine eigene Herleitung), RCL-Kappung bei wenigen Restknoten,
Zufallsabhängigkeit ab k>=2."""

import numpy as np
import pytest

import grasp_construct as GC
import grasp_tour as T


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return T.dist_matrix(xy)


@pytest.mark.parametrize("n", [6, 10, 20, 40])
def test_construct_is_a_valid_permutation_starting_at_the_depot(n):
    D = _instance(n, 1)
    rng = np.random.default_rng(0)
    for k in (1, 2, 5, n):
        t = GC.construct(D, k, rng)
        assert sorted(t.tolist()) == list(range(n)) and t[0] == 0


@pytest.mark.parametrize("n,seed", [(10, 0), (20, 1), (30, 2), (45, 3)])
def test_k_equals_one_matches_nearest_neighbor_tour_exactly(n, seed):
    D = _instance(n, seed)
    rng = np.random.default_rng(123)                              # bei k=1 hat der Zufall keinen Einfluss
    t = GC.construct(D, 1, rng)
    ref = T.nearest_neighbor_tour(D)
    assert np.array_equal(t, ref)


def test_k_equals_one_is_deterministic_across_different_rng_seeds():
    D = _instance(15, 4)
    a = GC.construct(D, 1, np.random.default_rng(1))
    b = GC.construct(D, 1, np.random.default_rng(999))
    assert np.array_equal(a, b)


def test_larger_k_produces_different_tours_across_rng_seeds():
    D = _instance(30, 5)
    tours = {tuple(GC.construct(D, 5, np.random.default_rng(s)).tolist()) for s in range(10)}
    assert len(tours) > 1                                          # mit Zufall entstehen tatsaechlich verschiedene Touren


def test_rcl_is_capped_when_fewer_nodes_remain_than_k():
    D = _instance(5, 6)                                             # nur 4 Knoten ausser dem Depot uebrig zu verteilen
    rng = np.random.default_rng(0)
    t = GC.construct(D, 20, rng)                                    # k weit groesser als n
    assert sorted(t.tolist()) == list(range(5))


def test_negative_or_zero_k_is_rejected():
    D = _instance(6, 0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        GC.construct(D, 0, rng)
