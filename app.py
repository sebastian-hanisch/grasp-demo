"""GRASP - eine Lieferrunde, viele randomisiert-gierige Neuanfänge - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe: direktes Kind von Hill Climbing, wie die fünf vorigen Stücke
auf derselben Lieferrunde. Anders als die anderen Geschwister (die alle von EINER Startlösung aus suchen) startet GRASP bei JEDEM
Neustart neu: eine randomisiert-gierige Konstruktion (die `k` nächsten unbesuchten Knoten als Kandidatenliste, einer davon zufällig
gewählt - bei k=1 exakt der nächste-Nachbar-Algorithmus) gefolgt von einem normalen 2-opt-Abstieg. Viele unabhängige Durchläufe,
die beste Tour zählt. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import grasp_constants as C
import grasp_tour as T
from grasp_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, scaling_table, sweep, verdict
from grasp_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from grasp_visualization import build_budget, build_instance, build_restarts, build_scaling, build_spread, build_sweep, build_tour, build_trace

st.set_page_config(page_title="GRASP – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🌱 GRASP – eine Lieferrunde, viele randomisiert-gierige Neuanfänge")
st.markdown(
    """
Alle bisherigen Geschwister-Demos suchen von **einer** Startlösung aus weiter. GRASP (Greedy Randomized Adaptive Search Procedure) macht
etwas anderes: es **konstruiert** bei jedem Neustart eine neue Tour – an jedem Schritt werden die `k` nächsten unbesuchten Stopps als
Kandidatenliste gebildet und **einer davon zufällig** gewählt (statt immer der nächste) –, löst sie mit einem normalen 2-opt-Abstieg, und
merkt sich die beste je gefundene Tour über viele solche unabhängige Durchläufe. Bei `k=1` ist die Kandidatenliste immer genau ein Knoten:
die Konstruktion ist dann exakt der **nächste-Nachbar-Algorithmus** – deterministisch, und Neustarts bringen nichts.
"""
)
st.caption(
    "Sechstes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe: dieselbe Rundtour wie in der "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/), der "
    "[simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/), der "
    "[iterated-local-search-demo](https://github.com/sebastian-hanisch/iterated-local-search-demo), der "
    "[variable-neighborhood-search-demo](https://github.com/sebastian-hanisch/variable-neighborhood-search-demo) und der "
    "[tabu-search-demo](https://github.com/sebastian-hanisch/tabu-search-demo) - ein Depot in der Mitte, n Kundenstopps in einem 100 × 100-km-Gebiet, euklidische Entfernungen. "
    "Der Nachbarschafts-Zweig (Lin-Kernighan, VLSN, VRP-Nachbarschaften) und ALNS sind andere Antworten auf dieselbe Schwäche der Wurzel."
)

with st.expander("So funktioniert GRASP", expanded=True):
    st.markdown(
        """
1. **Randomisiert-gierige Konstruktion.** Beginnend am Depot: an jedem Schritt werden die `k` nächsten unbesuchten Stopps als Restricted Candidate List (RCL) gebildet, einer davon wird **zufällig** gewählt (nicht immer der nächste).
2. **2-opt-Abstieg.** Die konstruierte Tour wird wie bei Hill Climbing bis zu einem lokalen Optimum verbessert (volle Nachbarschaft, beste Verbesserung).
3. **Neustart.** Schritt 1-2 wiederholen, bis das Budget aufgebraucht ist; die beste je gefundene Tour zählt.
4. **Grenzfall k=1.** Die RCL enthält dann immer nur den nächsten Knoten - die Konstruktion ist deterministisch (exakt `nearest_neighbor_tour`), jeder Neustart konstruiert identisch, Neustarts sind wirkungslos.
5. **Bewertung.** Der Abstand zur **1-Baum-Schranke**, wie in den Geschwister-Demos. Verglichen wird mit **Hill Climbing mit rein zufälligen Neustarts** (voller Rescan, gleiches Budget) und einem einzelnen **Abstieg aus dem nächsten Nachbarn**.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu). Je größer die Instanz, desto teurer ein einzelner Neustart - bei 200 Stopps und 1 Million Vorschlägen passt nur noch ein einziger Neustart hinein.",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    rcl = st.slider(
        "RCL-Größe k", C.RCL_MIN, C.RCL_MAX, key="rcl_slider",
        help="Wie viele der nächsten Stopps als Kandidaten für den nächsten Konstruktionsschritt in Frage kommen. k=1: rein gierig, deterministisch, Neustarts wirkungslos (7.30 % über der Schranke bei 200 Tausend). "
             "k=2-3: Sweet Spot (**4.34-4.35 %**, Voreinstellung k=3). Ab k≈10-20 nähert sich die Konstruktion einer zufälligen Startlösung an - kaum schlechter (4.64 % bei k=20), aber mit deutlich mehr Streuung.",
    )
    budget = st.select_slider(
        "Budget (bewertete Nachbarn)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Bei 10-50 Tausend passt nur ein Neustart, GRASP gewinnt trotzdem klar gegen einen zufälligen Start (6.93 % gegen 7.88 %) - die Konstruktion selbst ist besser. Der Vorteil ist am größten bei "
             "100-200 Tausend (5.43/4.34 % gegen 7.88/4.88 %) - bei sehr großem Budget (500 Tausend+) holt reiner Zufall durch schiere Menge an Neustarts auf und liegt bei 500 Tausend und 2 Millionen leicht vorn (2.02 % gegen 1.87 % bei 2 Millionen; bei 1 Million noch knapp dahinter: 2.46 % gegen 2.54 %).",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert die Zufallsfolge der Konstruktion (welcher RCL-Kandidat je Schritt gewählt wird) und die Startlösungen des Vergleichs (Neustarts, rein zufällig).",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "rcl_slider": int(rcl),
    "budget_select": int(budget), "chain_seed_input": int(chain_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), int(rcl), int(budget), int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
run = a.run
xy = a.inst.xy
code = verdict(a)
data_key = settings

# --- GRASP in Aktion -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 GRASP in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Neustarts", 3: "3 · Ergebnis"}
if "grasp_step" not in st.session_state or st.session_state.get("grasp_step_owner") != data_key:
    st.session_state["grasp_step"] = 1
    st.session_state["grasp_step_owner"] = data_key
    st.session_state.pop("grasp_restart", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="grasp_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

n_snaps = len(run.snapshots)
restart_idx = n_snaps - 1
play_search = False
if step == 2 and n_snaps > 1:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        restart_idx = st.slider("Neustart", 0, n_snaps - 1, value=n_snaps - 1, key="grasp_restart", help="Die beste Tour bisher nach diesem Neustart (0 = erster Neustart).")
    with itplay_col:
        play_search = st.button("▶️ Neustarts abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_snaps <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_snaps - 1, min(n_snaps, 40))})


def _render(current_step, it):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            st.markdown("**Beste Tour bisher über die bewerteten Nachbarn (alle Neustarts zusammen)**")
            st.plotly_chart(build_trace(run.trace_iter, run.trace_best, a.bound, a.nn.length, T.tour_length(a.hcr_tour, a.D)), width="stretch", key=f"s2_trace_{it}")
            st.markdown("**Länge jedes einzelnen Neustarts** (Streuung der randomisierten Konstruktion)")
            st.plotly_chart(build_restarts(run.restart_lengths, a.bound, run.best_length), width="stretch", key=f"s2_restarts_{it}")
            st.markdown(f"**Beste Tour bisher nach Neustart {it + 1} von {n_snaps}**")
            st.plotly_chart(build_tour(xy, run.snapshots[it], ghost=run.best_tour if it < n_snaps - 1 else None), width="stretch", key=f"s2_map_{it}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**GRASP: beste Tour** – {a.gap:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, run.best_tour), width="stretch", key="s3_grasp")
            c2.markdown(f"**Neustarts, rein zufällig** ({a.hcr_starts} Abstiege, gleiches Budget) – {a.hcr_gap:.1f} % über der Schranke")
            c2.plotly_chart(build_tour(xy, a.hcr_tour), width="stretch", key="s3_hcr")


if auto_play:
    for s in STEP_LABELS:
        if s == 2:
            for f in _frames():
                _render(2, f)
                time.sleep(0.1)
            time.sleep(0.6)
        else:
            _render(s, restart_idx)
            time.sleep(1.2)
    step = 3
elif play_search:
    for f in _frames():
        _render(2, f)
        time.sleep(0.1)
else:
    _render(step, restart_idx)

if step == 1:
    st.caption(f"{a.inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {a.bound:,.0f} km (1-Baum-Schranke, Held-Karp).".replace(",", "."))
elif step == 2:
    st.caption(f"{_fmt_int(run.evaluations)} bewertete Nachbarn in {run.starts} Neustarts (RCL-Größe {run.rcl}) - jeder Neustart konstruiert unabhängig und steigt bis zu einem lokalen Optimum ab.")
else:
    st.caption(f"Links die beste Tour aus {run.starts} GRASP-Neustarts, rechts die beste Tour aus {a.hcr_starts} rein zufälligen Neustarts mit demselben Bewertungsbudget ({_fmt_int(settings.budget)}).")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Suche gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. "
    "Ein Lauf ist eine Ziehung (die Konstruktion UND die Vergleichs-Neustarts sind randomisiert): Vergleiche gelten für diesen Lauf, andere Ketten streuen."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("GRASP: beste Tour", f"{a.gap:.1f} %", delta=f"{run.starts} Neustarts", delta_color="off", help="Abstand zur Schranke der besten je gefundenen Tour über alle Neustarts.")
m2.metric("Nächster Nachbar, ein Abstieg", f"{a.nn_gap:.1f} %", delta=f"{_fmt_int(a.nn.evaluations)} Bewertungen", delta_color="off", help="Ein Abstieg aus der rein gierigen Konstruktion (RCL-Größe 1), kein Neustart.")
m3.metric("Neustarts, rein zufällig", f"{a.hcr_gap:.1f} %", delta=f"{a.hcr_starts} Abstiege, gleiches Budget", delta_color="off", help="So viele Abstiege aus zufälligen Startlösungen, wie ins Budget passen; die beste Tour zählt.")
m4.metric("Streuung der Neustarts", f"{a.restart_gap_mean:.1f} ± {a.restart_gap_sd:.1f} %", delta="Mittel ± Streuung je Neustart", delta_color="off", help="Mittel und Streuung des Abstands zur Schranke über ALLE einzelnen Neustarts dieser Kette (nicht nur den besten).")

if code == "beats_hc":
    st.success(f"✅ Besser als Hill Climbing mit rein zufälligen Neustarts (gleiches Budget): {a.gap:.1f} % über der Schranke gegen {a.hcr_gap:.1f} % ({a.hcr_starts} Abstiege). Die informierte Konstruktion zahlt sich aus. Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: GRASP {a.gap:.1f} %, Neustarts rein zufällig {a.hcr_gap:.1f} % über der Schranke. Eine andere Kette kann das Bild drehen.")
else:
    st.warning(f"⚠️ Rein zufällige Neustarts sind besser: {a.hcr_gap:.1f} % gegen {a.gap:.1f} % über der Schranke bei gleichem Budget. Bei zu kleinem oder sehr großem Budget verschwindet der Vorteil der informierten Konstruktion - siehe README 'Was nicht funktioniert hat'.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Nachbarn", "Rechenzeit"],
              "GRASP (beste Tour)": [f"{run.best_length:.1f}", f"{a.gap:.2f} %", _fmt_int(run.evaluations), unit_time(a.seconds)],
              "Nächster Nachbar": [f"{a.nn.length:.1f}", f"{a.nn_gap:.2f} %", _fmt_int(a.nn.evaluations), unit_time(a.nn_seconds)],
              "Neustarts, rein zufällig": [f"{T.tour_length(a.hcr_tour, a.D):.1f}", f"{a.hcr_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.hcr_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["RCL-Größe k", "Budget", "Neustarts", "Bewertungen je Neustart", "Kreuzungen der besten Tour"],
              "Einstellung": [f"{settings.rcl}", _fmt_int(settings.budget), f"{run.starts}", f"{run.evaluations // max(run.starts, 1)}", f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag ist ein bewerteter Nachbar im 2-opt-Abstieg, dieselbe Einheit wie in den Geschwister-Demos - die Konstruktion selbst zählt nicht zum Budget. Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget, RCL-Größe und Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 20 bis 90 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Gepunktet: Neustarts, rein zufällig; gestrichelt: nächster Nachbar, ein Abstieg.")

st.markdown("---")

# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Budget: wann hilft die informierte Konstruktion?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 90 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
    st.plotly_chart(build_budget(rows_b), width="stretch", key="budget_chart")
    st.table({"Budget": [_fmt_int(r["value"]) for r in rows_b], "GRASP (%)": [f"{r['gap']:.2f}" for r in rows_b],
              "Neustarts, zufällig (%)": [f"{r['hcr']:.2f}" for r in rows_b], "nächster Nachbar (%)": [f"{r['nn']:.2f}" for r in rows_b], "Neustarts": [f"{r['starts']:.1f}" for r in rows_b]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (60 Stopps, RCL-Größe 3). Bei 10-50 Tausend passt nur ein Neustart - GRASP gewinnt trotzdem klar gegen einen zufälligen Start (**6.93 %** gegen **7.88 %**), "
               "die Konstruktion selbst ist besser. Der Vorteil ist am größten bei 100-200 Tausend (**5.43/4.34 %** gegen **7.88/4.88 %**). Bei sehr großem Budget (500 Tausend+) holt reiner Zufall durch schiere Menge an Neustarts auf und liegt bei 500 Tausend und 2 Millionen leicht vorn (**2.02 %** gegen **1.87 %** bei 2 Millionen; bei 1 Million noch knapp dahinter: **2.46 %** gegen **2.54 %**).")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 15 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["grasp"], sp["hcr"]), width="stretch", key="spread_chart")
    s1, s2 = st.columns(2)
    s1.metric("GRASP: Mittel ± Streuung", f"{sp['grasp'].mean():.2f} ± {sp['grasp'].std():.2f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten.")
    s2.metric("Neustarts, rein zufällig: Mittel ± Streuung", f"{sp['hcr'].mean():.2f} ± {sp['hcr'].std():.2f} %", help="Dieselben 20 Ketten, Hill Climbing mit rein zufälligen Neustarts statt GRASP.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (steuern sowohl die GRASP-Konstruktion als auch die zufälligen Vergleichs-Neustarts).")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 90 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget). Bei fester 200-Tausend-Grenze schrumpft die Zahl der Neustarts mit wachsendem n - "
               "bei 100-150 Stopps passt kaum mehr als ein Neustart hinein, GRASPs Vorteil verschwindet (siehe RCL-Befund: bei großem n gewinnt sogar k=1).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Budget für mehr als einen Neustart** | Bei 10-50 Tausend Vorschlägen (60 Stopps) passt nur EIN Neustart: das Ergebnis bleibt bei **6.93 %** stehen (kein Fortschritt durch Wiederholung) - schlägt aber trotzdem einen zufälligen Start (**7.88 %**) durch die bessere Konstruktion selbst. Erst ab 100 Tausend (mehrere Neustarts) verbessert sich das Ergebnis weiter (**5.43 %**). | Kleineres n, größeres Budget, oder ein billigeres Verfahren (**Kandidatenliste**: ILS/VNS) |
| **Die RCL-Größe passt zur Instanzgröße** | Bei kleinem n (20-30): k=1 verliert dramatisch (2.51/5.54 % gegen 0.05/0.57 % bei k=2) - Neustarts sind ohne Zufall wirkungslos. Bei großem n (100-150) DREHT sich das Bild: k=1 gewinnt (7.20/7.45 % gegen 9.03-9.91 % bei k≥2) - kaum Budget je Neustart übrig, ein vollständiger gieriger Abstieg schlägt mehrere abgebrochene randomisierte. | Kein direkter Nachfolger; dieselbe Lehre wie SA's Temperatur, ILS' Störstärke, VNS' k_max, Tabu Search's Tenure - aber hier zusätzlich n-abhängig |
| **Nicht zu viel Budget** | Ab 500 Tausend holt reiner Zufall durch schiere Menge an Neustarts auf und liegt bei 500 Tausend (**3.32 %** gegen **2.93 %**) und bei 2 Millionen (**2.02 %** gegen **1.87 %**) leicht vorn, bei 1 Million knapp dahinter (**2.46 %** gegen **2.54 %**) - der Vorteil der informierten Konstruktion ist ein MITTLERES Budgetfenster, kein durchgehender. | (kein Nachfolger nötig - bei genug Neustarts verschwimmt der Unterschied zwischen Verfahren) |
| **Jeder Neustart ist unabhängig** | GRASP wirft nach jedem Neustart die Tour komplett weg und konstruiert neu - anders als ILS/VNS, die den vorherigen Fortschritt gezielt stören statt zu verwerfen. Das kostet bei großem n Budget, das andere Verfahren gezielter einsetzen. | **ILS/VNS** (Kick statt Neukonstruktion) |
"""
)
st.caption(
    "Die Nachbarn der Trajektorien-Metaheuristiken-Linie: der Nachbarschafts-Zweig (Lin-Kernighan, VLSN, VRP-Nachbarschaften) ist eine "
    "andere Antwort auf dieselbe Schwäche der Wurzel; ALNS braucht eine CVRP-Instanz und steht deshalb in einer eigenen Demo ([alns-demo](https://github.com/sebastian-hanisch/alns-demo))."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**Konstruktion.** Beginnend am Depot $\pi_0 = 0$: an jedem Schritt $t$ sei $R_t$ die Menge der unbesuchten Knoten, sortiert nach Abstand vom zuletzt
besuchten Knoten $\pi_{t-1}$. Die Restricted Candidate List ist $\text{RCL}_t = $ die $k$ Knoten aus $R_t$ mit dem kleinsten Abstand zu $\pi_{t-1}$
(bei $|R_t| < k$: alle). Der nächste Knoten $\pi_t$ wird **gleichverteilt zufällig** aus $\text{RCL}_t$ gewählt. Bei $k=1$ ist
$\text{RCL}_t$ stets einelementig: die Konstruktion ist deterministisch und entspricht exakt dem nächste-Nachbar-Algorithmus.

**Lokale Suche.** Ein 2-opt-Abstieg (volle Nachbarschaft, beste Verbesserung) bis zu einem lokalen Optimum, wie bei Hill Climbing.

**Schleife.** Konstruktion → Abstieg → beste Tour merken, wiederholen, bis das Bewertungsbudget erschöpft ist. Die Konstruktion selbst zählt
nicht zum Budget, nur die Bewertungen im 2-opt-Abstieg (dieselbe Konvention wie der Zufallsstart bei Hill Climbing mit Neustarts).

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$ mit der 1-Baum-Schranke $w$. Vergleichsgrößen bei gleichem Budget: ein Abstieg
aus dem nächsten Nachbarn (RCL-Größe 1, kein Neustart) und Hill Climbing mit rein zufälligen Neustarts (voller Rescan, wie in der Wurzel-Demo).

**Grenzen.** (1) Die RCL-Größe ist ein Sweet-Spot-Parameter, der zusätzlich von der Zahl der ins Budget passenden Neustarts abhängt (also
von der Instanzgröße). (2) Der Vorteil gegenüber reinem Zufall ist auf ein mittleres Budgetfenster begrenzt. (3) Jeder Neustart verwirft
den vorherigen Fortschritt komplett - anders als ILS/VNS.

**Literatur.** Feo, T. A., & Resende, M. G. C. (1989). *A probabilistic heuristic for a computationally difficult set covering problem.*
Operations Research Letters, 8(2), 67-71. Feo, T. A., & Resende, M. G. C. (1995). *Greedy randomized adaptive search procedures.*
Journal of Global Optimization, 6(2), 109-133.

Implementiert in `grasp_construct.py` (die randomisierte gierige Konstruktion), `grasp_algorithm.py` (die Neustart-Schleife),
`grasp_tour.py` (Nachbarschaften, Abstieg, Schranke - aus der Hill-Climbing-Demo), `grasp_scenario.py` (Instanzen),
`grasp_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Trajektorien-Metaheuristiken: HC bis ALNS](https://sebastianhanisch.net/konzepte-trajektorien-metaheuristiken.html)."
)
