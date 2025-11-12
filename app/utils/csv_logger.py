# app/utils/csv_logger.py
import csv
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
CSV_PATH = LOG_DIR / "diagnosis_results.csv"

# Champs normalisés incluant les nouveaux ('expected_note', 'f0_seed_note')
HEADERS = [
    "timestamp",
    "expected_note",
    "expected_freq",
    "f0_seed_note",
    "librosa_f0", "librosa_conf",
    "essentia_f0", "essentia_conf",
    "hps_f0", "hps_q",
    "hpsm_f0", "hpsm_q",
    "final_f0", "final_conf",
    "delta_cents",
    "B",
    "n_cordes", "beat_hz",
    "valid",
    "t_total_ms",
]

# ──────────────────────────────────────────────
# MAIN LOGGER FUNCTION
# ──────────────────────────────────────────────
def append_diagnosis_row(row: dict):
    """Ajoute une ligne dans diagnosis_results.csv (crée le fichier si inexistant)."""

    # Crée le dossier logs si besoin
    CSV_PATH.parent.mkdir(exist_ok=True)
    write_header = not CSV_PATH.exists()

    # On ne garde que les clés valides, pour éviter les erreurs
    filtered_row = {k: row.get(k, "") for k in HEADERS}

    # Ajout du timestamp (si non fourni)
    if not filtered_row.get("timestamp"):
        filtered_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ───── Écriture sécurisée ─────
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=HEADERS,
            lineterminator="\n",
            extrasaction="ignore"   # ignore les clés non reconnues
        )
        if write_header:
            writer.writeheader()
        writer.writerow(filtered_row)