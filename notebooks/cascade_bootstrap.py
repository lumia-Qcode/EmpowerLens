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
import threading
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

# NOTE: deliberately NO Hugging Face environment manipulation here.
#
# An earlier version set HF_HUB_OFFLINE=1 (and download timeouts) to stop
# `from_pretrained` making a network call on every seed. That was a misdiagnosis
# and it CAUSED a failure that did not exist before: the pre-cache used an
# allow_patterns allowlist, so any file it did not match was never downloaded —
# and offline mode then made that file unfetchable, so AutoTokenizer.from_pretrained
# stalled instead of failing cleanly. Stage 1 trained fine before it was added.
#
# Leave `from_pretrained` to manage its own cache. If a genuine Hub stall ever
# needs handling, do it with a timeout around the call, not by pre-emptively
# blocking the network.

# --- config -----------------------------------------------------------------
MODEL = "mental/mental-roberta-base"
TAG = MODEL.split("/")[-1]
SEEDS = (42, 1337, 2024)

# ONE recipe, used by Stage 1, Stage 2, the flat comparator and the multiclass
# track alike. This is what makes Experiment 7 answerable: the two arms differ in
# ARCHITECTURE and nothing else.
#
# Previously each arm carried different flags - Stage 2 had focal loss, LLRD,
# lr 3e-5 and a cosine schedule while the flat comparator got bare defaults and
# a different epoch count. A flat-vs-cascade difference measured under those
# conditions cannot be attributed to the architecture, which is the only thing
# Experiment 7 is trying to measure.
#
#   512 tokens   truncates 2.4% of test rows; 256 truncated 24.9%
#   12 epochs    4 was never validated; 8 was the suite's budget; 12 gives the
#                peak more room. Every run saves epoch_history.csv, so check the
#                best epoch afterwards - if runs still peak at 12 of 12, raise it.
#   bs 16        matches the experiment suite
#   head trunc   the parser default, and what Izza's suite uses (no --truncation
#                flag there either). At 512 only 2.4% of rows truncate at all, so
#                head vs head_tail moves almost nothing - the reason to match is
#                to keep one fewer axis varying between the tracks.
#   deterministic  same-seed runs have differed by 0.047 macro-F1 without it
#
# NOT identical to experiments/kaggle_runner_flat_experiments.ipynb, and it does
# not need to be - the ONE comparison that must stay inside a single track is
# flat vs cascade, and both of those arms are here. Three things differ from
# Izza's E6, so E6 is not a replication of the flat arm below and the two must
# never be subtracted from each other:
#
#     E6 (Track A)                  E7 here (Track B)
#     8 epochs                      12 epochs
#     --loss weighted_bce           bce (plain; src/train_transformer.py's
#                                   --loss offers only {bce, focal}, so
#                                   weighted_bce is not reachable from here)
#     experiments_flat_...py        src/train_transformer.py
#   early stopping  OFF (patience 0), which is what every run in results_RUN2/
#                   used. src/train_transformer.py supports
#                   --early-stopping-patience, but turning it on here would
#                   make new Stage 1/2/flat numbers differ in provenance from
#                   the ones already in the tables. With load_best_model_at_end
#                   the kept weights are identical either way - patience only
#                   stops paying for the epochs after the peak. Set the env var
#                   EMPOWERLENS_EARLY_STOPPING=3 before the exec to enable it,
#                   and re-run every arm if you do.
RECIPE = "--max-length 512 --batch-size 16 --epochs 12 --deterministic"

_ESP = int(os.environ.get("EMPOWERLENS_EARLY_STOPPING", "0"))
if _ESP > 0:
    RECIPE += f" --early-stopping-patience {_ESP}"

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
#
# Switchable WITHOUT editing this file: set EMPOWERLENS_SPLITS before the exec.
#     import os; os.environ["EMPOWERLENS_SPLITS"] = "data/splits_codipas_cls"
#     exec(open(BOOT).read())
# It must be set BEFORE, not after — every path below is derived at exec time, so
# reassigning PARENT_SPLITS afterwards would leave STAGE2_SPLITS and the output
# dirs still pointing at the previous dataset.
PARENT_SPLITS = os.environ.get("EMPOWERLENS_SPLITS", "data/splits")
COMBINED_SPLITS = PARENT_SPLITS               # back-compat alias for older cells

# ---- everything below is DERIVED from PARENT_SPLITS -------------------------
# Change PARENT_SPLITS alone (e.g. to "data/splits_codipas_cls") and every path
# follows, so two datasets can never overwrite each other.
#
# Without this, running the cascade on CODIPAS would have:
#   * OVERWRITTEN data/splits_stage2 (Step 1 regenerates it in place), and
#   * written its results into results_stage1/ etc. on top of the Annotated run.
# The default ("splits") keeps the original unsuffixed names so the committed
# 2026-08-16 results stay where they are.
_DS = Path(PARENT_SPLITS).name                          # "splits" | "splits_codipas_cls"
_SUF = "" if _DS == "splits" else "_" + _DS.replace("splits_", "")

# Distorted-only, derived from PARENT_SPLITS by Step 1.
#
# NOT "data/splits_stage2" for the default dataset. That path is COMMITTED and was
# derived from data/splits_combined, so it carries the 396-row leak (77% of the
# Annotated test set). Writing over it in the clone would destroy the provenance of
# every older Stage 2 result and leave no way to tell a regenerated dir from the
# contaminated original. docs/E7_PROTOCOL.md names splits_stage2_annotated for
# exactly this reason; keep the two dirs distinct.
_STAGE2_SUF = "_annotated" if _SUF == "" else _SUF
STAGE2_SPLITS = f"data/splits_stage2{_STAGE2_SUF}"

# Every result this bootstrap writes lands under RUN2. RUN1 is the frozen
# pre-rerun history (no determinism, some leaked splits); a rerun must never
# land on top of numbers the thesis already cites. Override only if you know why.
RUN_ROOT = os.environ.get("EMPOWERLENS_RUN_ROOT", "results_RUN2")

STAGE1_OUT = f"{RUN_ROOT}/results_stage1{_SUF}"
MULTICLASS_OUT = f"{RUN_ROOT}/results_multiclass_v2{_SUF}"
STAGE2_OUT = f"{RUN_ROOT}/results_stage2{_SUF}"
CASCADE_OUT = f"{RUN_ROOT}/results_cascade{_SUF}"
FLAT_OUT = f"{RUN_ROOT}/results_multilabel_flat{_SUF}"

# Where Track B's rows join Track A's comparable table. Izza's section 10 globs
# results_RUN2/results_experiments/exp*/two_exams.csv, so anything not written
# here is invisible to it no matter how the zips are merged.
# Defined in the bootstrap rather than in the cell that uses it so the zip and
# verify cells still know the path after a kernel restart.
E7_OUT = f"{RUN_ROOT}/results_experiments/exp7"

# Per-epoch curves. src/train_transformer.py writes epoch_history.csv into each
# CHECKPOINT dir, and checkpoints live outside the repo and are never zipped -
# so without copying them into a results dir, the evidence for "was 12 epochs
# enough" is destroyed when the session ends.
EPOCH_HIST_OUT = f"{E7_OUT}/epoch_history"

# Checkpoints live OUTSIDE the repo clone.
#
# Cell 1 runs `rm -rf /kaggle/working/empowerlens` to re-clone, and its comment
# claims that only deletes the code, "not your checkpoints!". That was false while
# checkpoints sat at empowerlens/checkpoints/ — re-running cell 1 wiped every
# trained model, and the cascade eval then had nothing to load. Keeping them at
# /kaggle/working/checkpoints makes that comment true and makes cell 1 safe to
# re-run mid-session.
CKPT_DIR = (f"/kaggle/working/checkpoints{_SUF}" if Path("/kaggle/working").is_dir()
            else f"checkpoints{_SUF}")
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

ALL_OUT = (STAGE1_OUT, MULTICLASS_OUT, STAGE2_OUT, CASCADE_OUT, FLAT_OUT,
           E7_OUT, EPOCH_HIST_OUT)

for _d in ALL_OUT:
    Path(_d).mkdir(parents=True, exist_ok=True)


# --- helpers ----------------------------------------------------------------
def sh(cmd, timeout=None):
    """Run a command with a HARD timeout; kill the whole process group if it hangs.

    Output is PIPED and drained by a reader thread. This is not cosmetic — it is
    the fix for the stalls that plagued earlier runs.

    The previous version passed no stdout/stderr, so the child inherited the
    kernel's file descriptors and wrote into a pipe with a fixed ~64KB OS buffer.
    Nothing drained that pipe while the parent sat blocked in proc.wait(), so once
    training had emitted 64KB (roughly the point of the "Loading weights" bar) the
    child blocked forever on write() and the parent blocked forever on the child.
    A textbook deadlock — and timing-dependent, which is why the same seed would
    train fine once and stall the next time, and why running the identical command
    with `!python ...` always worked: Jupyter drains as it goes.
    """
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
    )

    def _drain():
        for line in proc.stdout:
            print(line, end="", flush=True)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()

    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT after {timeout}s] killing process group: {cmd}", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        rc = -1
    reader.join(timeout=10)     # let the tail of the output land before returning
    if rc != 0:
        print(f"[FAILED] exit code {rc}: {cmd}", flush=True)
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


def run_and_report(task, splits_dir, out_dir, seed, extra_flags="", eval_flags="",
                   ckpt_dir=None):
    """Train one config, evaluate it, drop the optimizer state, sync, report.

    Skips entirely if the eval JSON already exists, so re-running after a restart
    does not redundantly retrain configs you already have.

    ``eval_flags`` exists for Stage 2: evaluating against a distorted-only splits
    dir requires --allow-distorted-only, because src/evaluate.py now refuses it by
    default. Those numbers are ISOLATED diagnostics, never cascade results.
    """
    # ckpt_dir separates checkpoints that would otherwise collide: names are
    # {task}_{TAG}_{seed}, so a flat multilabel model and the distorted-only
    # Stage 2 model at the same seed write to the SAME path and silently
    # overwrite each other. Pass a different ckpt_dir for any second run of the
    # same task.
    ckpt_root = ckpt_dir or CKPT_DIR
    ckpt = f"{ckpt_root}/{task}_{TAG}_{seed}"
    eval_json = Path(out_dir) / f"eval_{TAG}_{task}_{seed}.json"

    if eval_json.exists():
        print(f"  [seed {seed}] already completed (found {eval_json}) — skipping")
        show_test_metrics(out_dir, task, seed)
        return ckpt

    rc = sh(
        f"python -m src.train_transformer --task {task} --model {MODEL} --seed {seed} "
        f"--device auto --splits {splits_dir} --out {ckpt_root} {extra_flags}",
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


if os.environ.get("HF_HUB_OFFLINE") == "1":
    # A previous version of this file set this and it broke model loading. If it
    # is still set in the kernel from an earlier exec, clear it.
    del os.environ["HF_HUB_OFFLINE"]
    print("[bootstrap] cleared a stale HF_HUB_OFFLINE=1 from this kernel")

print(f"[bootstrap] cwd={Path.cwd()}  model={MODEL}  seeds={SEEDS}  ckpts={CKPT_DIR}")
