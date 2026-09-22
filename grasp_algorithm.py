"""GRASP (Greedy Randomized Adaptive Search Procedure, Feo & Resende 1989/1995) für eine Rundtour (TSP): viele
unabhängige Durchläufe aus randomisiert-gieriger Konstruktion (`grasp_construct.construct`, RCL-Größe k) + 2-opt-
Abstieg (voller Rescan, wie Hill Climbing); die beste je gefundene Tour zählt. Strukturell fast identisch zu einem
Hill-Climbing-Neustart mit voller Nachbarschaft, nur der Startgenerator ist informiert (gierig) statt rein zufällig.
Ein Vorschlag = eine bewertete Nachbarschaft im 2-opt-Abstieg (dieselbe Einheit wie in der ganzen Trajektorien-
Metaheuristiken-Linie); die Konstruktion selbst zählt nicht zum Budget (dieselbe Konvention wie der Zufallsstart bei
Hill Climbing mit Neustarts oder der Kick bei ILS/VNS)."""

from dataclasses import dataclass, field

import numpy as np

import grasp_construct as C
import grasp_tour as T


@dataclass
class Run:
    best_tour: np.ndarray
    best_length: float
    evaluations: int = 0
    starts: int = 0
    rcl: int = 5
    snapshots: list = field(default_factory=list)          # beste Tour NACH jedem Neustart (monoton fallend), nur bei keep_snapshots=True
    restart_lengths: list = field(default_factory=list)     # Laenge JEDES einzelnen Neustarts (nicht nur des besten) - zeigt die Streuung
    trace_iter: np.ndarray = None
    trace_best: np.ndarray = None


def restarts(D, k, budget, seed, keep_snapshots=True, trace_points=300):
    """Ein Lauf: konstruieren (RCL-Größe `k`) → 2-opt-Abstieg → beste Tour merken, bis das Bewertungsbudget
    `budget` erschöpft ist (der erste Neustart läuft immer zu Ende, weitere nur mit dem Rest des Budgets)."""
    rng = np.random.default_rng(seed)
    evaluations = starts = 0
    best_tour = best_length = None
    snapshots = []
    restart_lengths = []
    tr_it, tr_best = [], []

    while evaluations < budget or best_tour is None:
        remaining = None if best_tour is None else budget - evaluations
        start = C.construct(D, k, rng)
        r = T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=remaining)
        evaluations += r.evaluations
        starts += 1
        restart_lengths.append(r.length)
        if best_tour is None or r.length < best_length:
            best_length, best_tour = r.length, r.tour
        if keep_snapshots:
            snapshots.append(best_tour.copy())
        tr_it.append(evaluations)
        tr_best.append(best_length)

    # bei sehr vielen Neustarts (kleines k, kleine Instanz) auf `trace_points` Stuetzstellen ausduennen, ohne den letzten Punkt zu verlieren
    if len(tr_it) > trace_points:
        idx = sorted(set(np.linspace(0, len(tr_it) - 1, trace_points).astype(int).tolist()) | {len(tr_it) - 1})
        tr_it = [tr_it[i] for i in idx]
        tr_best = [tr_best[i] for i in idx]
    return Run(best_tour, best_length, evaluations, starts, k, snapshots, restart_lengths, np.array(tr_it), np.array(tr_best))
