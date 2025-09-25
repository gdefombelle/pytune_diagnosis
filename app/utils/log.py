# app/utils/cr.py
import re
import csv
from pathlib import Path

# Mapping MIDI → nom de note
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_to_note(midi: int) -> str:
    note = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{note}{octave}"

# --- Localisation du log ---
BASE_DIR = Path(__file__).resolve().parent
log_path = BASE_DIR / "diagnosis_log.txt"
out_path = BASE_DIR / "diagnosis_results.csv"

if not log_path.exists():
    raise FileNotFoundError(f"❌ Log file not found: {log_path}")

with open(log_path, "r") as f:
    log = f.read()

# --- Regex pour extraire infos ---
note_re = re.compile(r"Received note (\d+)")
fft_re = re.compile(r"Guess FFT: f0=([\d\.]+)Hz .*conf=([\d\.]+)")
pattern_re = re.compile(r"Guess Pattern: f0=([\d\.]+)Hz .*conf=([\d\.]+)")
diff_re = re.compile(r"↔ Diff vs YIN: ([\+\-]?\d+\.\d+)¢")

rows = []
current_midi = None
fft_data = None
pattern_data = None
diff_data = None

# --- Parsing du log ---
for line in log.splitlines():
    if "Received note" in line:
        # Sauvegarde de la ligne précédente
        if current_midi is not None:
            fft_vals = fft_data if fft_data else ["", ""]
            pattern_vals = pattern_data if pattern_data else ["", ""]
            diff_val = diff_data if diff_data else ""
            rows.append([current_midi, midi_to_note(current_midi), *fft_vals, *pattern_vals, diff_val])

        # Reset pour la nouvelle note
        m = note_re.search(line)
        current_midi = int(m.group(1)) if m else None
        fft_data = None
        pattern_data = None
        diff_data = None

    elif "Guess FFT" in line:
        m = fft_re.search(line)
        if m:
            fft_data = [m.group(1), m.group(2)]

    elif "Guess Pattern" in line:
        m = pattern_re.search(line)
        if m:
            pattern_data = [m.group(1), m.group(2)]

    elif "↔ Diff vs YIN" in line:
        m = diff_re.search(line)
        if m:
            diff_data = m.group(1)

# Dernière ligne
if current_midi is not None:
    fft_vals = fft_data if fft_data else ["", ""]
    pattern_vals = pattern_data if pattern_data else ["", ""]
    diff_val = diff_data if diff_data else ""
    rows.append([current_midi, midi_to_note(current_midi), *fft_vals, *pattern_vals, diff_val])

# --- Écriture CSV ---
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["MIDI", "Note", "FFT f0 (Hz)", "FFT conf", "Pattern f0 (Hz)", "Pattern conf", "Diff vs YIN (¢)"])
    writer.writerows(rows)

print(f"✅ Export terminé -> {out_path}")