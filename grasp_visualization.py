"""Plotly-Abbildungen der GRASP-Demo: Karten, Verlauf über die Neustarts, Streuung der einzelnen Neustarts, Budget-
Vergleich, Sweeps, Kettenstreuung, Skalierung. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import grasp_constants as C

TOUR_COLOR = "#4c78a8"
GRASP_COLOR = "#54a24b"
NN_COLOR = "#7f7f7f"
HCR_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _line_trace(xy, edges, color, name, dash=None, width=2.5, showlegend=True):
    x, y = [], []
    for a, b in edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_instance(xy):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=TOUR_COLOR, line=dict(width=1, color="white")), name="Stopps", hovertemplate="Stopp %{customdata}<extra></extra>", customdata=np.arange(1, len(xy))))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_tour(xy, tour, ghost=None):
    """Tour; `ghost` (optional) ist eine zweite Tour, die blass darunter gezeichnet wird (z. B. die beste Tour)."""
    fig = go.Figure()
    if ghost is not None:
        fig.add_trace(_line_trace(xy, _tour_edges_list(ghost), "#c9d6e6", "beste Tour", width=6))
    fig.add_trace(_line_trace(xy, _tour_edges_list(tour), TOUR_COLOR, "Tour"))
    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=6, color="white", line=dict(width=1.5, color=TOUR_COLOR)), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    return _map_layout(fig)


def build_trace(trace_iter, trace_best, bound, nn_length, hcr_length):
    """Länge der besten Tour über die bewerteten Nachbarn (alle Neustarts); Schranke, ein Abstieg aus dem
    nächsten Nachbarn und Hill Climbing mit rein zufälligen Neustarts als Linien."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trace_iter, y=trace_best, mode="lines+markers", line=dict(color=GRASP_COLOR, width=2.5), marker=dict(size=5), name="beste Tour"))
    fig.add_hline(y=bound, line=dict(color=NN_COLOR, dash="dot"), annotation_text="untere Schranke", annotation_position="bottom right")
    fig.add_hline(y=nn_length, line=dict(color=NN_COLOR, dash="dash"), annotation_text="nächster Nachbar, ein Abstieg", annotation_position="top right")
    fig.add_hline(y=hcr_length, line=dict(color=HCR_COLOR, dash="dash"), annotation_text="Neustarts, rein zufällig", annotation_position="top right")
    fig.update_xaxes(title_text="Bewertete Nachbarn")
    fig.update_yaxes(title_text="Länge (km)", range=[bound * 0.95, max(float(np.percentile(trace_best, 95)), nn_length * 1.05)])
    return _base(fig, 340)


def build_restarts(restart_lengths, bound, best_length):
    """Länge JEDES einzelnen Neustarts (Streuung der randomisierten Konstruktion) über den Neustart-Index."""
    xs = list(range(1, len(restart_lengths) + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=restart_lengths, marker_color=GRASP_COLOR, name="Neustart"))
    fig.add_hline(y=bound, line=dict(color=NN_COLOR, dash="dot"), annotation_text="untere Schranke", annotation_position="bottom right")
    fig.add_hline(y=best_length, line=dict(color="#e45756", dash="dash"), annotation_text="beste Tour", annotation_position="top right")
    fig.update_xaxes(title_text="Neustart")
    fig.update_yaxes(title_text="Länge nach 2-opt-Abstieg (km)", range=[bound * 0.95, max(restart_lengths) * 1.03])
    return _base(fig, 320)


def build_budget(rows):
    """Abstand zur Schranke über das Budget: GRASP gegen Hill Climbing mit rein zufälligen Neustarts und gegen einen Abstieg aus dem nächsten Nachbarn."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=GRASP_COLOR, width=2.5), name="GRASP"))
    fig.add_trace(go.Scatter(x=xs, y=[r["hcr"] for r in rows], mode="lines+markers", line=dict(color=HCR_COLOR, width=2.5), name="Neustarts, rein zufällig"))
    fig.add_trace(go.Scatter(x=xs, y=[r["nn"] for r in rows], mode="lines", line=dict(color=NN_COLOR, width=1.5, dash="dash"), name="nächster Nachbar, ein Abstieg"))
    fig.update_xaxes(title_text="Budget (bewertete Nachbarn)", type="log")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_sweep(rows, param_label):
    """Abstand zur Schranke (GRASP, Streuung als Band) und die Vergleichsgrößen über die Werte eines Reglers."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    upper = [r["gap"] + r["gap_sd"] for r in rows]
    lower = [max(0.0, r["gap"] - r["gap_sd"]) for r in rows]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(84,162,75,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=GRASP_COLOR, width=2.5), name="GRASP"))
    fig.add_trace(go.Scatter(x=xs, y=[r["hcr"] for r in rows], mode="lines+markers", line=dict(color=HCR_COLOR, width=2, dash="dot"), name="Neustarts, rein zufällig"))
    fig.add_trace(go.Scatter(x=xs, y=[r["nn"] for r in rows], mode="lines", line=dict(color=NN_COLOR, width=1.5, dash="dash"), name="nächster Nachbar, ein Abstieg"))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 380)


def build_spread(grasp, hcr):
    fig = go.Figure()
    fig.add_trace(go.Box(y=grasp, name="GRASP", marker_color=GRASP_COLOR, boxmean=True))
    fig.add_trace(go.Box(y=hcr, name="Neustarts, rein zufällig", marker_color=HCR_COLOR, boxmean=True))
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    return _base(fig, 360)


def build_scaling(blocks):
    """`blocks`: [{"label": ..., "rows": [...]}, ...] (siehe grasp_evaluation.scaling_table)."""
    fig = go.Figure()
    colors = (GRASP_COLOR, HCR_COLOR)
    for block, color in zip(blocks, colors):
        label, rows = block["label"], block["rows"]
        xs = [r["value"] for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=label))
    fig.update_xaxes(title_text="Stopps")
    fig.update_yaxes(title_text="Abstand zur Schranke (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)
