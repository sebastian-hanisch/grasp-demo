"""Konstanten der GRASP-Demo: Szenario (wortgleich zur hill-climbing-demo), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20

# --- GRASP-eigene Regler ------------------------------------------------------------------------------------------
RCL_MIN, RCL_MAX, DEFAULT_RCL = 1, 20, 3
DEFAULT_BUDGET = 200000

# --- Gemessene Werte (Mittel über 5 feste Sweep-Instanzen, Seeds 100000-100004, je 3 Ketten-Seeds - SWEEP_CHAINS; ---
# --- n=60, Anteil=0, sofern nicht anders angegeben; 2026-09-22) ------------------------------------------------------
# RCL-Sweep bei Budget 200 Tausend: k=1 (rein gierig) 7.30 %, k=2 4.35 %, k=3 4.34 %, k=5 4.99 %, k=10 5.83 %, k=20
#   4.64 % Abstand zur Schranke - Sweet Spot bei k=2-3 (k=1: Neustarts nutzlos, jeder identisch; groesseres k:
#   Konstruktion naehert sich Zufall an, verliert den gierigen Vorteil, k=10 zeigt das deutlichste Zwischentief).
#   k=3 gewaehlt: praktisch gleichauf mit k=2, robuster ueber n hinweg (bei n=30 sind k=2/k=3 praktisch gleich,
#   0.57/0.68 %).
# WICHTIGER, n-abhaengiger Befund: bei kleinem n (20-30) verliert k=1 dramatisch (2.51 %/5.54 % vs. 0.05 %/0.57 % bei
#   k=2) - viele Neustarts passen ins Budget, und ohne Zufall sind sie wirkungslos. Bei grossem n (100-150) DREHT
#   sich das Bild um: k=1 gewinnt (7.20/7.45 % vs. 9.19-9.91 % bei k=2-5) - pro Neustart bleibt kaum Budget uebrig,
#   ein einzelner vollstaendiger gieriger Abstieg schlaegt mehrere abgebrochene randomisierte. Der Sweet Spot
#   haengt also nicht an k allein, sondern daran, wie viele Neustarts das Budget je Instanzgroesse erlaubt.
# Hauptvergleich (k=3) gegen Hill Climbing mit rein zufaelligen Neustarts (HCR), ueber das Budget hinweg:
#   10-50 Tausend: 6.93 % vs. 7.88 % (nur 1-1.5 Neustarts passen ueberhaupt hinein, GRASP gewinnt trotzdem klar
#     schon beim ersten Versuch - die Konstruktion selbst ist besser als ein zufaelliger Start).
#   100 Tausend: 5.43 % vs. 7.88 % (GRASP gewinnt klar).
#   200 Tausend (Standard): 4.34 % vs. 4.88 % (GRASP gewinnt).
#   500 Tausend - 2 Millionen: 3.32/2.46/2.02 % vs. 2.93/2.54/1.87 % (HCR liegt hier leicht VORN - bei sehr vielen
#     Neustarts holt der reine Zufall durch schiere Menge auf und zieht leicht vorbei).
#   -> die informierte Konstruktion hilft am deutlichsten in einem MITTLEREN Budgetfenster (ca. 100-200 Tausend bei
#      n=60); bei sehr grossem Budget gleicht reiner Zufall den Vorteil aus und kann sogar knapp vorbeiziehen.
# Presets: Standard (rcl=3, 200T) 4.34 % gegen 4.88 % (HCR)/7.30 % (ein Abstieg); rcl=1 7.30 % (Neustarts nutzlos,
#   s.o.); rcl=20 4.64 % (fast so gut wie rcl=3, die Konstruktion naehert sich aber messbar dem Zufall an: mehr
#   Streuung zwischen den Neustarts); Budget 25T 6.93 % gegen 7.88 % (HCR) - nur 1 Neustart, GRASPs Konstruktion
#   hilft trotzdem; Budget 1M 2.46 % gegen 2.54 % (fast Gleichstand); grosse Instanz (200 Stopps, 1M) 8.60 % gegen
#   9.48 % (HCR)/8.84 % (ein Abstieg) - bei nur 1 Neustart gewinnt die gierige Konstruktion trotzdem knapp.


def _preset(rcl=DEFAULT_RCL, budget=DEFAULT_BUDGET, n=DEFAULT_N):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "rcl": rcl, "budget": budget, "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Rein gierig (RCL 1)": _preset(rcl=1),
    "Fast zufällig (RCL 20)": _preset(rcl=20),
    "Zu kleines Budget (25 Tausend)": _preset(budget=25000),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "Große Instanz (200 Stopps, 1 Million)": _preset(n=200, budget=1000000),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), Abstand zur Schranke; HCR = Hill Climbing mit rein zufälligen Neustarts bei gleichem Budget
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, RCL-Größe 3, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 4.34 % über der Schranke - Hill Climbing mit rein zufälligen Neustarts (gleiches Budget) 4.88 %, ein einzelner Abstieg aus dem nächsten Nachbarn 7.30 %.",
    "Rein gierig (RCL 1)": "Die RCL enthält immer nur den nächsten Knoten: jede Konstruktion ist identisch, Neustarts bringen nichts. 7.30 % über der Schranke, genauso viel wie ein einzelner Abstieg - trotz rund 14 (nutzloser) Neustarts.",
    "Fast zufällig (RCL 20)": "Die RCL umfasst 20 der nächsten Knoten: die Konstruktion nähert sich einer zufälligen Startlösung an. 4.64 % über der Schranke - kaum schlechter als die kalibrierte RCL-Größe 3 (4.34 %), aber mit deutlich mehr Streuung zwischen den Neustarts.",
    "Zu kleines Budget (25 Tausend)": "Nur 25 Tausend Vorschläge reichen für genau einen Neustart (wie bei Hill Climbing mit Neustarts auch): 6.93 % über der Schranke gegen 7.88 % bei rein zufälligem Start - schon der erste, informierte Konstruktionsversuch hilft.",
    "Großes Budget (1 Million)": "1 Million Vorschläge erlauben rund 21 Neustarts: 2.46 % über der Schranke - Hill Climbing mit rein zufälligen Neustarts liegt bei 2.54 %, praktisch gleichauf. Bei sehr großem Budget holt reiner Zufall die informierte Konstruktion durch schiere Menge fast ein.",
    "Große Instanz (200 Stopps, 1 Million)": "200 Stopps: ein einzelner 2-opt-Abstieg kostet bei dieser Größe fast das gesamte Budget, nur 1 Neustart passt hinein. 8.60 % über der Schranke gegen 9.48 % bei rein zufälligem Start - die gierige Konstruktion gewinnt auch ohne einen einzigen Neustart.",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x 2 Ketten gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_hc", "comparable", "hc_wins"},
    "Rein gierig (RCL 1)": {"hc_wins", "comparable", "beats_hc"},
    "Fast zufällig (RCL 20)": {"beats_hc", "comparable", "hc_wins"},
    "Zu kleines Budget (25 Tausend)": {"beats_hc", "comparable", "hc_wins"},
    "Großes Budget (1 Million)": {"beats_hc", "comparable", "hc_wins"},
    "Große Instanz (200 Stopps, 1 Million)": {"beats_hc", "comparable", "hc_wins"},
}
