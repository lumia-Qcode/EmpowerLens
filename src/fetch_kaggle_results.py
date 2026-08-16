"""
Pull a Kaggle notebook's output files straight into this repo, then compile them.

Replaces the manual zip-and-download dance that has already cost one full run:
`/kaggle/working` is wiped when a session ends, so results that were never
downloaded are simply gone.

One-time setup
--------------
    venv\\Scripts\\python.exe -m pip install kaggle
    venv\\Scripts\\kaggle.exe auth login

`auth login` opens a browser and caches credentials locally — no token file to
manage, and nothing secret ever passes through this script.

Usage
-----
    python -m src.fetch_kaggle_results --kernel <username>/<notebook-slug>

The slug is the tail of the notebook URL:
    https://www.kaggle.com/code/nayabshahbaz/cascade-run-v1
                                 ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^
                                 username       slug

Downloads into the repo root, so `results_*/` land where `src/compile_results.py`
expects them, then runs the compiler unless --no-compile is passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RESULT_DIRS = (
    "results_stage1", "results_stage2", "results_cascade",
    "results_multiclass_v2", "results_multilabel_flat",
)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch Kaggle notebook outputs into the repo.")
    ap.add_argument("--kernel", required=True,
                    help="<username>/<notebook-slug>, e.g. nayabshahbaz/cascade-run-v1")
    ap.add_argument("--dest", default=".", help="download target (default: repo root)")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip the src.compile_results step")
    args = ap.parse_args(argv)

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", args.kernel, "-p", str(dest)]
    print(f"$ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(
            "\nDownload failed. Most likely causes:\n"
            "  * not logged in      -> venv\\Scripts\\kaggle.exe auth login\n"
            "  * wrong slug         -> copy it from the notebook URL, after /code/\n"
            "  * notebook never saved a version -> Kaggle only exposes outputs from a\n"
            "    saved version, not from a live interactive session\n",
            file=sys.stderr,
        )
        return rc

    found = [d for d in RESULT_DIRS if (dest / d).is_dir() and any((dest / d).iterdir())]
    print(f"\nResult folders with content: {found or 'NONE'}")
    if not found:
        print(
            "  Nothing landed. If the notebook wrote to /kaggle/working/results_*, make\n"
            "  sure it ran via Save Version -> Save & Run All (Commit); outputs from a\n"
            "  purely interactive session are not retrievable after it ends."
        )
        return 1

    if not args.no_compile:
        print()
        subprocess.call([sys.executable, "-m", "src.compile_results"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
