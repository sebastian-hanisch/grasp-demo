"""Auswertung der GRASP-Demo: ein Lauf gegen Hill Climbing mit rein zufälligen Neustarts (voller Rescan, wie in der Wurzel-Demo)
und einen einzelnen Abstieg aus dem nächsten Nachbarn, gleiches Bewertungsbudget. Sweeps, Vergleichstabellen, Kettenstreuung.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour (1-Baum, Held-Karp). Ein Vorschlag ist ein
bewerteter Nachbar im 2-opt-Abstieg; die Konstruktion selbst (egal ob randomisiert-gierig oder rein zufällig) zählt nicht zum Budget."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import grasp_algorithm as G
import grasp_constants as C
import grasp_scenario as S
import grasp_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    rcl: int = C.DEFAULT_RCL
    budget: int = C.DEFAULT_BUDGET
    chain_seed: int = C.DEFAULT_CHAIN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    inst, D = instance(n, cluster_share, seed)
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return T.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


def hill_climbing_restarts(D, budget, seed):
    """Hill Climbing mit REIN ZUFÄLLIGEN Neustarts, voller Rescan (wie in der Wurzel-Demo) - die faire Vergleichsgröße:
    dieselbe lokale Suche wie GRASP, nur mit einem uninformierten statt gierig-randomisierten Startgenerator."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = T.descend(D, T.random_tour(len(D), rng), "2opt", "first", keep_steps=False, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best.length:
            best = r
    return best.tour, starts, used


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    bound: float
    run: object
    seconds: float
    nn: object                      # ein Abstieg aus dem nächsten Nachbarn (k=1, kein Neustart)
    nn_seconds: float
    hcr_tour: np.ndarray             # Hill Climbing mit rein zufälligen Neustarts, gleiches Budget
    hcr_starts: int
    hcr_seconds: float
    crossings_end: int

    def gap_of(self, length):
        return 100.0 * (length - self.bound) / self.bound

    @property
    def gap(self):
        return self.gap_of(self.run.best_length)

    @property
    def nn_gap(self):
        return self.gap_of(self.nn.length)

    @property
    def hcr_gap(self):
        return self.gap_of(T.tour_length(self.hcr_tour, self.D))

    @property
    def restart_gap_mean(self):
        return float(np.mean([self.gap_of(x) for x in self.run.restart_lengths]))

    @property
    def restart_gap_sd(self):
        return float(np.std([self.gap_of(x) for x in self.run.restart_lengths]))


def analyse(settings, keep_snapshots=True, with_hc=True):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    t0 = time.perf_counter()
    run = G.restarts(D, settings.rcl, settings.budget, settings.chain_seed, keep_snapshots=keep_snapshots)
    seconds = time.perf_counter() - t0
    nn = hcr_tour = None
    nn_seconds = hcr_seconds = 0.0
    hcr_starts = 0
    if with_hc:
        t0 = time.perf_counter()
        nn = T.descend(D, T.nearest_neighbor_tour(D), "2opt", "first", keep_steps=False)
        nn_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        hcr_tour, hcr_starts, _ = hill_climbing_restarts(D, settings.budget, settings.chain_seed)
        hcr_seconds = time.perf_counter() - t0
    return Analysis(settings, inst, D, bound, run, seconds, nn, nn_seconds, hcr_tour, hcr_starts, hcr_seconds,
                     T.count_crossings(inst.xy, run.best_tour))


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.3                  # so viel besser als Hill Climbing mit rein zufälligen Neustarts gilt als Sieg (Prozentpunkte)
LOSE_MARGIN = 0.3                 # so viel schlechter gilt als Niederlage


def verdict(a):
    """Code: beats_hc (deutlich besser als Hill Climbing mit rein zufälligen Neustarts bei gleichem Budget - die informierte
    Konstruktion hilft), hc_wins, comparable. Gilt für diesen einen Lauf - die Ketten streuen."""
    if a.gap <= a.hcr_gap - WIN_MARGIN:
        return "beats_hc"
    if a.hcr_gap <= a.gap - LOSE_MARGIN:
        return "hc_wins"
    return "comparable"


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch), keep_snapshots=False)
            rows.append({"gap": a.gap, "nn": a.nn_gap, "hcr": a.hcr_gap, "hcr_starts": a.hcr_starts, "seconds": a.seconds,
                         "starts": a.run.starts, "restart_gap_mean": a.restart_gap_mean, "restart_gap_sd": a.restart_gap_sd, "crossings": a.crossings_end})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"gap_sd": float(np.std([r["gap"] for r in rows])), "gap_min": float(np.min([r["gap"] for r in rows])),
                "gap_max": float(np.max([r["gap"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000), "rcl": (1, 2, 3, 5, 10, 20),
                "n": (10, 20, 40, 60, 100, 150, 200), "cluster_share": (0, 25, 50, 75, 100)}
SWEEP_LABELS = {"budget": "Budget (bewertete Nachbarn)", "rcl": "RCL-Größe k", "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


SCALING_N = C.SCALING_N
SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in SCALING_N]} for label, fn in SCALING_POLICIES]


def chain_spread(settings, k=C.SPREAD_CHAINS):
    """k Ketten-Seeds auf derselben Instanz: GRASP (beste Tour) gegen Hill Climbing mit rein zufälligen Neustarts,
    gleiches Budget - beide Verfahren sind randomisiert, der Vergleich zeigt, welches über die Ketten verlässlicher ist."""
    grasp_gaps, hcr_gaps = [], []
    for ch in range(k):
        a = analyse(replace(settings, chain_seed=ch), keep_snapshots=False)
        grasp_gaps.append(a.gap)
        hcr_gaps.append(a.hcr_gap)
    return {"grasp": np.array(grasp_gaps), "hcr": np.array(hcr_gaps)}
