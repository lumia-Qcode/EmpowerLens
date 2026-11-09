#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def read_workbooks(scripts_dir):
    """
    Reads all annotator workbook exports, excluding the master key, and searches
    the usual reannotation folders so the script can run regardless of where it is
    launched from.
    """
    base_dir = Path(scripts_dir).resolve()
    search_dirs = [
        base_dir,
        base_dir.parent,
        base_dir.parent / "outputs",
        base_dir.parent / "outputs" / "relabel",
        Path.cwd(),
        Path.cwd() / "reannotation_2026-09-27",
        Path.cwd() / "reannotation_2026-09-27" / "outputs" / "relabel",
    ]

    def is_annotator_workbook(path: Path) -> bool:
        name = path.name.lower()
        return (
            name.startswith("annotator_")
            and not name.startswith("annotator_master")
            and bool(re.fullmatch(r"annotator_\d+.*\.xlsx", name))
        )

    workbook_files = []
    seen = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for candidate in sorted(d.glob("annotator_*.xlsx")):
            if candidate in seen:
                continue
            seen.add(candidate)
            if is_annotator_workbook(candidate):
                workbook_files.append(candidate)

        if not workbook_files:
            for candidate in sorted(d.rglob("annotator_*.xlsx")):
                if candidate in seen:
                    continue
                seen.add(candidate)
                if is_annotator_workbook(candidate):
                    workbook_files.append(candidate)

        if workbook_files:
            break

    if not workbook_files:
        searched = ", ".join(str(d) for d in search_dirs if d.exists()) or str(base_dir)
        print(f"⚠️ No annotator workbook files found in any expected reannotation folder. Searched: {searched}")
        return {}, []

    all_labels = {}
    invalid_files = []

    for path in workbook_files:
        try:
            wb = load_workbook(path, data_only=True)
            ws = wb.active # Select the active/first worksheet safely
            
            # Extract and normalize headers: convert to lower, strip whitespace, remove internal spaces
            raw_headers = [cell.value for cell in ws[1]]
            normalized_headers = [
                str(h).strip().replace(" ", "").lower() if h is not None else "" 
                for h in raw_headers
            ]
            
            # Map normalized names to their column indexes
            header_map = {name: i for i, name in enumerate(normalized_headers)}
            
            # Verify absolute minimum dependency index exists
            if "rowid" not in header_map:
                print(f"❌ Missing 'Row ID' layout column in file: {path.name}")
                invalid_files.append(path.name)
                continue

            # Safely fetch indexes with fallbacks for optional columns
            row_id_idx = header_map["rowid"]
            primary_idx = header_map.get("primary")
            secondary_idx = header_map.get("secondary")
            unsure_idx = header_map.get("unsure")
            notes_idx = header_map.get("notes")

            file_entries = {}
            
            # Iterate through data rows skipping row 1 (the header)
            for r_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
                if row_id_idx >= len(row):
                    continue
                    
                raw_row_id = row[row_id_idx]
                if raw_row_id is None:
                    continue
                
                try:
                    row_id = int(float(str(raw_row_id).strip()))
                except ValueError:
                    # Skip rows where row ID isn't cleanly convertible to a numeric index
                    continue

                # Safely parse data items
                primary = None
                if primary_idx is not None and primary_idx < len(row) and row[primary_idx]:
                    primary = str(row[primary_idx]).strip() or None

                secondary = None
                if secondary_idx is not None and secondary_idx < len(row) and row[secondary_idx]:
                    secondary = str(row[secondary_idx]).strip() or None

                unsure = None
                if unsure_idx is not None and unsure_idx < len(row):
                    raw_unsure = row[unsure_idx]
                    if isinstance(raw_unsure, bool):
                        unsure = "Yes" if raw_unsure else None
                    else:
                        text = str(raw_unsure or "").strip().lower()
                        if text in {"true", "yes", "y", "x", "1", "☑"}:
                            unsure = "Yes"

                note = None
                if notes_idx is not None and notes_idx < len(row) and row[notes_idx]:
                    note = str(row[notes_idx]).strip() or None

                file_entries[row_id] = {
                    "Primary": primary,
                    "Secondary": secondary,
                    "Unsure": unsure,
                    "Notes": note
                }
                
            all_labels[path.name] = file_entries
            print(f"✅ Successfully parsed {len(file_entries)} entries from {path.name}")

        except Exception as e:
            print(f"❌ Error reading file {path.name}: {e}")
            invalid_files.append(path.name)

    return all_labels, invalid_files


def calculate_agreement(all_labels):
    """
    Computes inter-annotator percentage agreement metrics 
    across common Row IDs mapped from files.
    """
    if not all_labels:
        print("No label sets to cross-compare.")
        return

    file_names = list(all_labels.keys())
    if len(file_names) < 2:
        print("ℹ️ Found less than 2 files. Agreement metrics require multiple files to compare.")
        return

    # Aggregate common Row IDs across files
    common_ids = None
    for name in file_names:
        ids = set(all_labels[name].keys())
        if common_ids is None:
            common_ids = ids
        else:
            common_ids = common_ids.intersection(ids)

    if not common_ids:
        print("⚠️ No overlapping Row IDs found between these datasets to compute agreement statistics.")
        return

    print(f"\nEvaluating agreement on {len(common_ids)} intersecting rows...")
    
    # Calculate matrix matching comparisons pairs
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            f1, f2 = file_names[i], file_names[j]
            match_primary = 0
            match_secondary = 0
            match_unsure = 0
            
            for rid in common_ids:
                if all_labels[f1][rid]["Primary"] == all_labels[f2][rid]["Primary"]:
                    match_primary += 1
                if all_labels[f1][rid]["Secondary"] == all_labels[f2][rid]["Secondary"]:
                    match_secondary += 1
                if all_labels[f1][rid]["Unsure"] == all_labels[f2][rid]["Unsure"]:
                    match_unsure += 1
                    
            total = len(common_ids)
            print(f"\n📊 Agreement stats between [{f1}] and [{f2}]:")
            print(f"  - Primary Category Match   : {match_primary / total * 100:.2f}% ({match_primary}/{total})")
            print(f"  - Secondary Category Match : {match_secondary / total * 100:.2f}% ({match_secondary}/{total})")
            print(f"  - Unsure Flag Match        : {match_unsure / total * 100:.2f}% ({match_unsure}/{total})")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    print("🚀 Initializing data ingestion pipeline...")
    
    labels, invalid = read_workbooks(current_dir)
    
    if invalid:
        print(f"\n⚠️ Skiped {len(invalid)} files containing formatting structure problems: {invalid}")
        
    calculate_agreement(labels)
    print("\n🏁 Script task run finished successfully.")

if __name__ == "__main__":
    main()
