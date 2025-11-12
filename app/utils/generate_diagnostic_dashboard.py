# app/utils/generate_diagnostic_dashboard.py
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "logs" / "diagnosis_results.csv"
HTML_OUT = BASE_DIR / "logs" / "diagnosis_dashboard.html"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"❌ CSV introuvable : {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Nettoyage de base
df = df.sort_values("midi").reset_index(drop=True)
df["NoteLabel"] = df["note"] + " (" + df["midi"].astype(str) + ")"

# --- Création de la figure Plotly avec 3 sous-graphes ---
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "🎵 Écart d’accordage (Δ cents)",
        "🔧 Inharmonicité (B)",
        "⚡ Temps de traitement total (ms)"
    ],
    vertical_spacing=0.1
)

# 1️⃣ Δ cents
fig.add_trace(
    go.Scatter(
        x=df["midi"], y=df["delta_cents"],
        mode="markers+lines",
        name="Δ cents",
        marker=dict(color="cyan", size=6),
        hovertext=df["NoteLabel"]
    ),
    row=1, col=1
)

# 2️⃣ B (inharmonicité)
fig.add_trace(
    go.Scatter(
        x=df["midi"], y=df["B"],
        mode="markers+lines",
        name="Inharmonicité B",
        marker=dict(color="orange", size=6),
        hovertext=df["NoteLabel"]
    ),
    row=2, col=1
)

# 3️⃣ Temps total
fig.add_trace(
    go.Scatter(
        x=df["midi"], y=df["t_total_ms"],
        mode="markers+lines",
        name="Temps total (ms)",
        marker=dict(color="magenta", size=6),
        hovertext=df["NoteLabel"]
    ),
    row=3, col=1
)

# --- Mise en forme globale ---
fig.update_layout(
    title=dict(
        text="🎹 Tableau de bord diagnostic PyTune",
        font=dict(size=24, color="white"),
        x=0.5
    ),
    showlegend=False,
    template="plotly_dark",
    height=1000,
    margin=dict(l=60, r=40, t=80, b=60),
)

fig.update_xaxes(title="Note MIDI")
fig.update_yaxes(title="Δ (cents)", row=1, col=1)
fig.update_yaxes(title="B", row=2, col=1)
fig.update_yaxes(title="Temps (ms)", row=3, col=1)

fig.write_html(HTML_OUT, include_plotlyjs="cdn", full_html=True)
print(f"✅ Dashboard généré : {HTML_OUT}")