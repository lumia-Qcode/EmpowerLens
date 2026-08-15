"""Restart-proof state for notebooks/kaggle_runner_cascade.ipynb.

WHY THIS FILE EXISTS
--------------------
Every config value and helper used to live in notebook cells. Kaggle kernels
restart — 12-hour session limit, OOM, manual restart — and when they do, every
Python variable and the working directory are lost. Running a later cell without
re-running the earlier ones then fails with, literally:

    NameError: name 'STAGE2_SPLITS' is not defined

which is what killed Stage 2 training on the last real run, and with it the
cascade evaluation that depends on Stage 2's checkpoints.

Putting the state on disk instead of in kernel memory makes it survivable: every
stage cell starts with the same idempotent line

    exec(open(BOOT).read())

so cells can be run in any order, after any restart, without a NameError.

Safe to exec repeatedly — it only sets names and creates directories.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

# --- locate the repo and cd into it -----------------------------------------
# A kernel restart also resets the working directory, so never assume we are
# already inside the clone.
# NOTE: this file is loaded with exec(), so __file__ is NOT defined here — locate
# the repo by looking for a marker file instead.
def _find_repo():
    kaggle = Path("/kaggle/working/empowerlens")
    if (kaggle / "src" / "data.py").exists():
        return kaggle
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):         # running locally from anywhere in the tree
        if (cand / "src" / "data.py").exists():
            return cand
    raise RuntimeError(
        "Cannot locate the EmpowerLens repo (no src/data.py found in /kaggle/working/"
        f"empowerlens or above {here}). Re-run the clone cell first."
    )


REPO_DIR = _find_repo()
os.chdir(REPO_DIR)

# Pin to a single GPU BEFORE any subprocess imports torch. Kaggle sometimes
# assigns T4 x2 and the multi-device weight-loading path can deadlock silently.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- config -----------------------------------------------------------------
MODEL = "mental/mental-roberta-base"
TAG = MODEL.split("/")[-1]
SEEDS = (42, 1337, 2024)

TRAIN_TIMEOUT = 3600
EVAL_TIMEOUT = 3600

# PARENT_SPLITS must be a FULL splits dir (containing No-Distortion rows).
#
# Currently data/splits — the original Annotated_data.csv splits, verified clean:
# train-in-val = 0, train-in-test = 0.
#
# NOT data/splits_combined: CODIPAS overlaps Annotated_data heavily (1,937 of its
# 2,621 rows are already in it), so merging leaked 194/253 val rows and 189/253
# test rows into train — 75% of the evaluation set. Any result from that dir is
# invalid. Switch back only after CODIPAS is deduplicated against the frozen
# Annotated val/test.
PARENT_SPLITS = "data/splits"
COMBINED_SPLITS = PARENT_SPLITS               # back-compat alias for older cells
STAGE2_SPLITS = "data/splits_stage2"          # distorted-only, derived from PARENT_SPLITS

STAGE1_OUT = "results_stage1"
MULTICLASS_OUT = "results_multiclass_v2"
STAGE2_OUT = "results_stage2"
CASCADE_OUT = "results_cascade"

for _d in (STAGE1_OUT, MULTICLASS_OUT, STAGE2_OUT, CASCADE_OUT):
    Path(_d).mkdir(parents=True, exist_ok=True)


# --- helpers ----------------------------------------------------------------
def sh(cmd, timeout=None):
    """Run a command with a HARD timeout; kill the whole process group if it hangs."""
    print(f"$ {cmd}")
    proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT after {timeout}s] killing process group: {cmd}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        rc = -1
    if rc != 0:
        print(f"[FAILED] exit code {rc}: {cmd}")
    return rc


def sync(folder):
    """Persist a results folder to /kaggle/working so it survives the session."""
    dest = Path(f"/kaggle/working/{folder}")
    dest.mkdir(parents=True, exist_ok=True)
    sh(f"cp -r {folder}/* {dest}/")


def show_test_metrics(out_dir, task, seed):
    p = Path(out_dir) / f"eval_{TAG}_{task}_{seed}.json"
    if not p.exists():
        print(f"  [seed {seed}] eval JSON not found at {p} — train/evaluate failed or timed out")
        return
    m = json.loads(p.read_text())["splits"]["test"]["metrics"]
    if task == "binary":
        print(f"  [seed {seed}] test weighted_f1={m['weighted_f1']:.3f}  "
              f"positive_class_f1={m['positive_class_f1']:.3f}")
    elif task == "multiclass":
        print(f"  [seed {seed}] test weighted_f1={m['weighted_f1']:.3f}  "
              f"macro_f1_10={m['macro_f1_10']:.3f}")
    else:
        print(f"  [seed {seed}] test weighted_f1={m['weighted_f1']:.3f}  "
              f"macro_f1={m['macro_f1']:.3f}")


def run_and_report(task, splits_dir, out_dir, seed, extra_flags="", eval_flags=""):
    """Train one config, evaluate it, drop the optimizer state, sync, report.

    Skips entirely if the eval JSON already exists, so re-running after a restart
    does not redundantly retrain configs you already have.

    ``eval_flags`` exists for Stage 2: evaluating against a distorted-only splits
    dir requires --allow-distorted-only, because src/evaluate.py now refuses it by
    default. Those numbers are ISOLATED diagnostics, never cascade results.
    """
    ckpt = f"checkpoints/{task}_{TAG}_{seed}"
    eval_json = Path(out_dir) / f"eval_{TAG}_{task}_{seed}.json"

    if eval_json.exists():
        print(f"  [seed {seed}] already completed (found {eval_json}) — skipping")
        show_test_metrics(out_dir, task, seed)
        return ckpt

    rc = sh(
        f"python -m src.train_transformer --task {task} --model {MODEL} --seed {seed} "
        f"--device auto --splits {splits_dir} {extra_flags}",
        timeout=TRAIN_TIMEOUT,
    )

    if rc == 0:
        print("Waiting 15 seconds for GPU VRAM to flush...")
        time.sleep(15)
        sh(
            f"python -m src.evaluate --checkpoint {ckpt} --reference "
            f"--splits {splits_dir} --out {out_dir} {eval_flags}",
            timeout=EVAL_TIMEOUT,
        )

    # The Trainer's checkpoint-XXX/ holds optimizer+scheduler state that evaluate.py
    # never reads. Remove it regardless of success — it is the main disk hog.
    sh(f"rm -rf {ckpt}/checkpoint-*")
    show_test_metrics(out_dir, task, seed)
    sync(out_dir)
    return ckpt


print(f"[bootstrap] cwd={Path.cwd()}  model={MODEL}  seeds={SEEDS}")
