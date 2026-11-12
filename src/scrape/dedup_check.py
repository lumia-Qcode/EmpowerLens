"""S01 check — rows per platform before/after exact de-duplication, against the 489 target.

Reads every data/supplementary/raw/*_raw.csv (Izza's files are picked up automatically).
Read-only: it reports, it does not rewrite the raw files.

    venv\\Scripts\\python.exe -m src.scrape.dedup_check
"""
from __future__ import annotations

import pandas as pd

from src.scrape.common import COLUMNS, RAW_DIR

TARGET = 489  # docs/T16_target_dataset_sizes.xlsx


def main() -> None:
    files = sorted(RAW_DIR.glob("*_raw.csv"))
    if not files:
        raise SystemExit(f"no *_raw.csv files in {RAW_DIR}")
    df = pd.concat([pd.read_csv(f, encoding="utf-8", dtype=str) for f in files], ignore_index=True)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    norm = df["text"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    dedup = df[~norm.duplicated()]
    table = pd.DataFrame({"raw": df.platform.value_counts(), "after_exact_dedup": dedup.platform.value_counts(),
                          "threads": dedup.groupby("platform").thread_id.nunique()}).fillna(0).astype(int)
    print(table.to_string())
    print(f"\ntotal raw {len(df)}, after exact dedup {len(dedup)} "
          f"({len(df) - len(dedup)} duplicates); target {TARGET}: "
          f"{'MET' if len(dedup) >= TARGET else f'short by {TARGET - len(dedup)}'}")


if __name__ == "__main__":
    main()
