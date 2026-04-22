import argparse
import csv
from pathlib import Path


CRITICAL_COLUMNS = [
    "played_note",
    "chosen_note",
    "chosen_note_2",
    "chosen_note_3",
    "chosen_note_4",
    "chosen_method",
    "chosen_method_2",
    "chosen_method_3",
    "chosen_method_4",
]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize(value):
    if value is None:
        return None
    v = str(value).strip()
    return v if v != "" else None


def compare_rows(ref_rows: list[dict], cur_rows: list[dict]) -> list[str]:
    diffs: list[str] = []

    if len(ref_rows) != len(cur_rows):
        diffs.append(
            f"Nombre de lignes différent: ref={len(ref_rows)} courant={len(cur_rows)}"
        )
        return diffs

    for i, (ref_row, cur_row) in enumerate(zip(ref_rows, cur_rows), start=1):
        for col in CRITICAL_COLUMNS:
            ref_val = normalize(ref_row.get(col))
            cur_val = normalize(cur_row.get(col))
            if ref_val != cur_val:
                diffs.append(
                    f"Ligne {i} colonne '{col}': ref={ref_val!r} courant={cur_val!r}"
                )

    return diffs


def main():
    parser = argparse.ArgumentParser(
        description="Compare deux CSV de replay f0 sur les colonnes critiques."
    )
    parser.add_argument("--reference", required=True, help="CSV de référence")
    parser.add_argument("--current", required=True, help="CSV courant à vérifier")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    cur_path = Path(args.current)

    if not ref_path.exists():
        raise FileNotFoundError(f"CSV de référence introuvable: {ref_path}")
    if not cur_path.exists():
        raise FileNotFoundError(f"CSV courant introuvable: {cur_path}")

    ref_rows = load_csv(ref_path)
    cur_rows = load_csv(cur_path)

    diffs = compare_rows(ref_rows, cur_rows)

    if diffs:
        print("❌ Régression détectée")
        print()
        for diff in diffs[:50]:
            print(diff)
        if len(diffs) > 50:
            print()
            print(f"... {len(diffs) - 50} différences supplémentaires")
        raise SystemExit(1)

    print("✅ Aucune régression sur les colonnes critiques")


if __name__ == "__main__":
    main()