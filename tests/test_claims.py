"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen (je drei Ketten-Seeds) belegt.
Positive UND negative Aussagen: GRASP schlägt reinen Zufall nur in einem mittleren Budgetfenster, und der RCL-Sweet-Spot dreht sich mit der
Instanzgröße um - das steht hier ebenso als Test wie die Stellen, an denen die informierte Konstruktion klar gewinnt."""

from functools import lru_cache

import pytest

import grasp_constants as C
import grasp_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap"], 4.34, 1.5)
    near(std["hcr"], 4.88, 1.5)
    near(std["nn"], 7.30, 1.0)


# --- RCL-Sweep: Sweet Spot bei k=2-3 -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rcl,gap,tol", [(1, 7.30, 1.5), (2, 4.35, 1.5), (3, 4.34, 1.5), (5, 4.99, 1.5), (10, 5.83, 2.0), (20, 4.64, 1.5)])
def test_rcl_sweep_numbers_at_two_hundred_thousand(rcl, gap, tol):
    near(cfg(rcl=rcl)["gap"], gap, tol)


def test_rcl_one_makes_restarts_pointless_and_is_worse_than_the_sweet_spot():
    """Bei RCL-Größe 1 ist jeder Neustart (bis auf den letzten, durch das Restbudget abgeschnittenen) identisch -
    die Streuung je Neustart bleibt deutlich unter der bei RCL-Größe 3 (siehe test_algorithm.py für den exakten Beweis)."""
    one = cfg(rcl=1)
    three = cfg(rcl=3)
    assert one["restart_gap_sd"] < three["restart_gap_sd"]
    assert three["gap"] < one["gap"] - 1.0


def test_rcl_twenty_is_close_to_the_sweet_spot_but_more_spread_out():
    three = cfg(rcl=3)
    twenty = cfg(rcl=20)
    assert twenty["gap"] < three["gap"] + 1.0                        # kaum schlechter
    assert twenty["restart_gap_sd"] > three["restart_gap_sd"]        # aber deutlich mehr Streuung je Neustart


# --- Der n-abhängige RCL-Befund: der Sweet Spot dreht sich um -----------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _cfg_n(n, rcl):
    return ev.run_config(ev.Settings(n=n, budget=200000), rcl=rcl)


@pytest.mark.parametrize("n,gap_k1,gap_k2,tol", [(20, 2.51, 0.05, 1.5), (30, 5.54, 0.57, 2.0)])
def test_small_instances_make_rcl_one_lose_dramatically_to_rcl_two(n, gap_k1, gap_k2, tol):
    near(_cfg_n(n, 1)["gap"], gap_k1, tol)
    near(_cfg_n(n, 2)["gap"], gap_k2, tol)
    assert _cfg_n(n, 1)["gap"] > _cfg_n(n, 2)["gap"] + 1.0


@pytest.mark.parametrize("n,gap_k1,tol", [(100, 7.20, 1.5), (150, 7.45, 1.5)])
def test_large_instances_flip_the_picture_rcl_one_wins(n, gap_k1, tol):
    near(_cfg_n(n, 1)["gap"], gap_k1, tol)
    for rcl in (2, 3, 5):
        assert _cfg_n(n, 1)["gap"] < _cfg_n(n, rcl)["gap"] - 0.5


# --- Budget-Sweep: das mittlere Fenster -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget,gap,tol", [
    (10000, 6.93, 3.0), (25000, 6.93, 3.0), (50000, 6.93, 3.0), (100000, 5.43, 2.5),
    (200000, 4.34, 1.5), (500000, 3.32, 1.5), (1000000, 2.46, 1.2), (2000000, 2.02, 1.2),
])
def test_budget_sweep_numbers(budget, gap, tol):
    near(cfg(budget=budget)["gap"], gap, tol)


def test_grasp_beats_hcr_at_small_budgets_via_the_construction_itself():
    for budget in (10000, 25000, 50000):
        row = cfg(budget=budget)
        assert row["starts"] < 2.0                                   # nur ein Neustart passt hinein
        assert row["gap"] < row["hcr"] - 0.3                         # trotzdem besser: die Konstruktion selbst hilft


def test_grasp_wins_clearly_in_the_middle_budget_window():
    mid = cfg(budget=200000)
    assert mid["gap"] < mid["hcr"] - 0.3


def test_pure_random_restarts_catch_up_and_can_edge_ahead_at_very_large_budgets():
    mid = cfg(budget=200000)
    huge = cfg(budget=2000000)
    mid_edge = mid["hcr"] - mid["gap"]
    huge_edge = huge["hcr"] - huge["gap"]
    assert mid_edge > 0 and huge_edge < mid_edge                     # der Vorsprung schrumpft (und kann sich hier sogar umdrehen)


# --- Große Instanz ----------------------------------------------------------------------------------------------------------------------------


def test_large_instance_only_one_restart_fits_but_the_construction_still_wins():
    row = ev.run_config(ev.Settings(n=200), budget=1000000)
    near(row["gap"], 8.60, 2.5)
    near(row["hcr"], 9.48, 2.5)
    assert row["starts"] < 2.0
    assert row["gap"] < row["hcr"] - 0.3


# --- Sonstiges --------------------------------------------------------------------------------------------------------------------------------


def test_bound_matches_the_frozen_reference():
    near(ev.reference_bound(60, 0, 100000), 618.76, 0.1)


def test_preset_count_matches_the_readme():
    assert len(C.PRESETS) == 6
