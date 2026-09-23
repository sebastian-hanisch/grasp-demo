# GRASP – eine Lieferrunde, viele randomisiert-gierige Neuanfänge – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-grasp-demo.streamlit.app/)**

Sechstes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
dieselbe Rundtour wie in der [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/), der [simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/), der [iterated-local-search-demo](https://sebastianhanisch-iterated-local-search-demo.streamlit.app/), der [variable-neighborhood-search-demo](https://sebastianhanisch-variable-neighborhood-search-demo.streamlit.app/) und der [tabu-search-demo](https://sebastianhanisch-tabu-search-demo.streamlit.app/) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselbe untere Schranke.

**Einordnung in die Reihe:** **GRASP** (Greedy Randomized Adaptive Search Procedure, Feo & Resende 1989/1995) ist ein direktes Kind von Hill Climbing, wie die fünf vorigen Stücke - aber es sucht nicht von EINER Startlösung aus weiter. Jeder Neustart **konstruiert** eine neue Tour: an jedem Schritt werden die `k` nächsten unbesuchten Stopps als Restricted Candidate List (RCL) gebildet, einer davon wird **zufällig** gewählt (statt immer der nächste - das wäre `nearest_neighbor_tour`, exakt der Grenzfall k=1). Danach folgt ein normaler 2-opt-Abstieg. Viele unabhängige Konstruktion-plus-Abstieg-Durchläufe, die beste Tour zählt - strukturell fast identisch zu Hill Climbing mit Neustarts, nur mit einem **informierten statt rein zufälligen** Startgenerator.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [gebaut]
  ├─ iterated-local-search-demo (stört ein gutes Optimum mit fester Störstärke)  [gebaut]
  │     └─ variable-neighborhood-search-demo (Störstärke eskaliert + Reset)     [gebaut]
  ├─ tabu-search-demo (immer der beste Zug, Gedächtnis gegen Rückwege)          [gebaut]
  └─ grasp-demo (randomisierte Konstruktion, viele Neuanfänge)                  [dieses Stück]
```

Ergebnis in Kürze: **eine informierte Konstruktion schlägt reinen Zufall bei gleichem Budget - aber nur in einem mittleren Budgetfenster, und die kalibrierte RCL-Größe hängt von der Instanzgröße ab.** 60 Stopps, 200 Tausend Vorschläge (Standard): GRASP (RCL-Größe 3) liegt bei **4.34 %** über der Schranke gegen **4.88 %** für Hill Climbing mit rein zufälligen Neustarts (gleiches Budget) und **7.30 %** für einen einzelnen Abstieg aus dem nächsten Nachbarn. Der Vorsprung ist am deutlichsten zwischen 100 Tausend und 200 Tausend Vorschlägen; bei sehr großem Budget (500 Tausend+) holt reiner Zufall durch schiere Menge an Neustarts auf und zieht bei 2 Millionen sogar knapp vorbei (2.02 % gegen 1.87 %).
Die **RCL-Größe** hat einen Sweet Spot bei k=2-3 - aber dieser Sweet Spot ist n-abhängig: bei kleinen Instanzen (20-30 Stopps, viele Neustarts passen ins Budget) verliert k=1 dramatisch (2.51 %/5.54 % gegen 0.05 %/0.57 % bei k=2), bei großen Instanzen (100-150 Stopps, kaum Budget je Neustart) DREHT sich das Bild und k=1 gewinnt (7.20/7.45 % gegen 9.03-9.91 % bei k≥2).

| Frage | Ergebnis (60 gleichverteilte Stopps, RCL-Größe 3, Budget 200 Tausend; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke) |
|---|---|
| Standardfall | ➖ GRASP **4.34 %** über der Schranke gegen **4.88 %** für Hill Climbing mit rein zufälligen Neustarts und **7.30 %** für einen Abstieg aus dem nächsten Nachbarn |
| **RCL-Größe k** | ⚠️ 1 / 2 / 3 / 5 / 10 / 20: **7.30 / 4.35 / 4.34 / 4.99 / 5.83 / 4.64 %** - Sweet Spot bei 2-3, aber n-abhängig (s. u.) |
| **RCL bei kleiner Instanz (30 Stopps)** | ❌ k=1: **5.54 %**, k=2: **0.57 %** - Neustarts sind ohne Zufall wirkungslos |
| **RCL bei großer Instanz (150 Stopps)** | ✅ k=1: **7.45 %** schlägt k=2-5 (**9.03-9.91 %**) - kaum Budget je Neustart, ein vollständiger gieriger Abstieg gewinnt |
| **Budget** | ⚠️ Bei 10-50 Tausend Vorschlägen: **6.93 %** (nur 1 Neustart, GRASP gewinnt trotzdem gegen HCR **7.88 %**). Bei 100 / 200 Tausend: **5.43 / 4.34 %** (GRASP klar besser). Bei 500 Tausend / 1 / 2 Millionen: **3.32 / 2.46 / 2.02 %** gegen HCR **2.93 / 2.54 / 1.87 %** - hier liegt reiner Zufall leicht VORN |
| **Größe** | ❌ 200 Stopps, 1 Million Vorschläge (nur 1 Neustart): **8.60 %** gegen **9.48 %** für Hill Climbing mit Neustarts - die gierige Konstruktion gewinnt auch ohne Neustart-Vorteil |

## Was die Demo zeigt

1. **GRASP in Aktion** (Schritt-Slider + Abspielen): **Instanz** → **Neustarts** (Neustart-Regler + ▶️ Neustarts abspielen: beste Tour bisher über die bewerteten Nachbarn, Länge jedes einzelnen Neustarts als Balken, Tour nach dem gewählten Neustart) → **Ergebnis** (beste Tour neben der besten aus rein zufälligen Neustarts).
2. **Was die Suche gefunden hat:** beste Tour, nächster Nachbar, Neustarts rein zufällig, Streuung der einzelnen Neustarts; Urteil (`beats_hc` → `comparable` → `hc_wins`), Detailtabellen.
3. **📐 Sweeps** über Budget, RCL-Größe, Stopps und Gruppen (feste Instanzen ab 100000, drei Ketten je Instanz).
4. **🔬 Experimente auf Abruf:** Budget von 10 Tausend bis 2 Millionen (die "mittleres Fenster"-Geschichte); Streuung über 20 Ketten; Skalierung von 20 bis 200 Stopps.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (genug Budget für mehr als einen Neustart, die RCL-Größe passt zur Instanzgröße, nicht zu viel Budget, jeder Neustart ist unabhängig).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **RCL-Größe k** (1–20), **Budget** (10 Tausend bis 2 Millionen bewertete Nachbarn), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲).
Kein Startlösung-Regler (anders als bei den Geschwister-Demos - jede GRASP-Iteration konstruiert ihre eigene randomisierte Startlösung neu, es gibt keine einzelne feste Kette); kein Nachbarschafts-Regler (bewusst nur 2-opt, wie Tabu Search).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

| Preset | Urteil (Band über Instanzen × Ketten) |
|---|---|
| Standardfall (Voreinstellung) | beats_hc / comparable / hc_wins |
| Rein gierig (RCL 1) | beats_hc / comparable / hc_wins |
| Fast zufällig (RCL 20) | beats_hc / comparable / hc_wins |
| Zu kleines Budget (25 Tausend) | beats_hc / comparable / hc_wins |
| Großes Budget (1 Million) | beats_hc / comparable / hc_wins |
| Große Instanz (200 Stopps, 1 Million) | beats_hc / comparable / hc_wins |

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke** (`grasp_scenario.py`, `grasp_tour.py`): wortgleiche Kopien aus der [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) (per Test gegen eingefrorene Werte) - GRASP braucht dieselbe volle, vektorisierte 2-opt-Bewertung wie Tabu Search, keine Kandidatenliste/Doppelbrücke der ILS/VNS-Familie.
- **Randomisierte gierige Konstruktion** (`grasp_construct.py`): direkte Verallgemeinerung von `nearest_neighbor_tour` - an jedem Schritt die `k` nächsten unbesuchten Knoten als RCL, einer davon zufällig gewählt. Bei k=1 exakt `nearest_neighbor_tour` (Regressionstest gegen die bereits geprüfte Funktion, keine eigene Herleitung).
- **Neustart-Schleife** (`grasp_algorithm.py`): konstruieren → 2-opt-Abstieg (voller Rescan, Restbudget begrenzt) → beste Tour merken, wiederholen bis das Budget erschöpft ist. Die Konstruktion selbst zählt nicht zum Budget (dieselbe Konvention wie der Zufallsstart bei Hill Climbing mit Neustarts).
- **Hill Climbing mit rein zufälligen Neustarts** (`grasp_evaluation.py`): dieselbe Schleife, aber mit einem uninformierten Zufallsstart statt der randomisierten Konstruktion - die faire Vergleichsgröße bei gleichem Budget.
- **Auswertung** (`grasp_evaluation.py`): Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Streuung, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "eine informierte Konstruktion schlägt reinen Zufall bei gleichem Budget"** – **bestätigt, aber mit einem eng begrenzten Fenster**: der Vorteil ist am deutlichsten zwischen 100 Tausend und 200 Tausend Vorschlägen (5.43-4.34 % gegen 7.88-4.88 %). Bei sehr kleinem Budget (10-50 Tausend) passt nur EIN Neustart, dort gewinnt GRASP trotzdem schon durch die bessere Konstruktion selbst. Bei sehr großem Budget (500 Tausend+) holt reiner Zufall durch die schiere Menge an Neustarts auf und zieht bei 2 Millionen sogar knapp vorbei (2.02 % gegen 1.87 %) - der Vorsprung ist ein Fenster, kein Dauerzustand, dieselbe Lehre wie bei Tabu Search's Budget-Story, nur in die andere Richtung (dort half mehr Budget durchgehend, hier verpufft der Vorteil bei zu viel Budget).
- **Die RCL-Größe ist ein Sweet-Spot-Parameter, der zusätzlich von der Instanzgröße abhängt** - anders als bei SA/ILS/VNS/Tabu Search, wo der Sweet Spot bei fester Instanzgröße stabil war. Bei kleinem n (20-30, viele Neustarts passen ins Budget) verliert k=1 dramatisch (2.51 %/5.54 % gegen 0.05 %/0.57 % bei k=2) - ohne Zufall ist jeder Neustart identisch, also wirkungslos. Bei großem n (100-150, kaum Budget je Neustart) DREHT sich das Bild komplett: k=1 gewinnt (7.20/7.45 % gegen 9.03-9.91 % bei k≥2) - ein einzelner vollständiger gieriger Abstieg schlägt mehrere randomisierte, die durch das Restbudget abgebrochen werden. Der "richtige" k-Wert hängt also nicht an der Instanz selbst, sondern daran, wie viele Neustarts das Budget bei dieser Größe überhaupt erlaubt.
- **Bei k=1 sind Neustarts nachweislich wirkungslos** (außer dem letzten, budget-abgeschnittenen): die Konstruktion ist deterministisch, jeder Neustart liefert exakt dieselbe Tour nach demselben 2-opt-Abstieg - ein sauberer, getesteter Grenzfall.
- **Jeder Neustart verwirft den vorherigen Fortschritt komplett** - anders als ILS/VNS, die eine gute Lösung gezielt stören statt neu zu konstruieren. Bei großer Instanzgröße (200 Stopps) kostet das: nur 1 Neustart passt in 1 Million Vorschläge, die informierte Konstruktion gewinnt dort nur, weil sie besser startet, nicht weil sie mehrfach probiert.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Konstruktion:** bei RCL-Größe 1 liefert `grasp_construct.construct` für jeden getesteten Seed exakt dieselbe Tour wie `nearest_neighbor_tour` (Regressionstest, keine eigene Herleitung); die RCL wird korrekt auf die Zahl der übrigen Knoten gekappt, wenn k größer als der Rest ist.
- **k=1-Grenzfall auf Algorithmusebene:** bei RCL-Größe 1 sind alle Neustarts (bis auf den letzten, durch das Restbudget abgeschnittenen) exakt identisch - unabhängig nachgerechnet über die rohen `restart_lengths`, nicht nur über den Endwert.
- **Budget-Buchführung:** der erste Neustart läuft bei jedem Budget immer vollständig zu Ende (auch bei einem winzigen Budget); jeder weitere Neustart wird auf das Restbudget begrenzt; mehr Budget verschlechtert die beste Tour nie.
- Übernommener Kern: 2-opt gegen Brute-Force, Abstieg strikt monoton und im lokalen Optimum, Bewertungsbudget, 1-Baum-Schranke gegen Brute-Force (n = 8) und CP-SAT (n = 20); Instanz gegen eingefrorene Werte.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, RCL-, Budget- und Größen-Aussagen; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung); alle 6 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 10 und 60 Stopps, Neustart-Regler, ▶️ Abspielen und ▶️ Neustarts abspielen ohne doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Budget, Streuung, Skalierung), 🚧 Grenzen, Mathe |
| `grasp_construct.py` | Die randomisierte gierige Konstruktion (Restricted Candidate List) |
| `grasp_algorithm.py` | Die Neustart-Schleife: konstruieren, 2-opt-Abstieg, beste Tour merken |
| `grasp_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke (aus der Hill-Climbing-Demo) |
| `grasp_scenario.py`, `grasp_constants.py` | Instanzen; Konstanten, Presets |
| `grasp_evaluation.py` | Analyse, Urteil, Hill Climbing mit rein zufälligen Neustarts, Sweeps, Streuung, Skalierung |
| `grasp_presets.py`, `grasp_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Übernommener Kern, Konstruktion (Regressionstest gegen nächster Nachbar), Neustart-Schleife, Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
