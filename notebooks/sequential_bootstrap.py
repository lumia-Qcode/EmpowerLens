"""Restart-proof state for notebooks/kaggle_runner_sequential.ipynb.

Layered on top of cascade_bootstrap.py: execs it first (for sh(), sync(), the
timeouts, the GPU pin and the repo cd), then adds what sequential fine-tuning
needs on top.

WHY A SEPARATE FILE
-------------------
Same reason cascade_bootstrap.py exists. A Kaggle kernel restart wipes every
Python variable, so a helper defined in a notebook cell disappears and later
cells fail with `NameError: name 'run_stage' is not defined`. Keeping it on disk
means every cell can begin with the same idempotent line

    exec(open(SEQBOOT).read())

and be run in any order, after any restart.

WHY NOT EXTEND run_and_report
-----------------------------
The bootstrap's run_and_report hardcodes `--model {MODEL}`, and stage B has to
initialise from stage A's checkpoint instead. Rather than change a function the
cascade notebook depends on, this defines its own runner.

Safe to exec repeatedly.
"""

import json
import os
import time
from pathlib import Path

# --- load the shared bootstrap ----------------------------------------------
# Locate it relative to the repo, not to __file__ — this file is exec()'d too, so
# __file__ is not defined here either.
_CASCADE = Path("/kaggle/working/empowerlens/notebooks/cascade_bootstrap.py")
if not _CASCADE.exists():
    for _c in (Path.cwd(), *Path.cwd().parents):
        if (_c / "notebooks" / "cascade_bootstrap.py").exists():
            _CASCADE = _c / "notebooks" / "cascade_bootstrap.py"
            break
exec(open(_CASCADE).read())

# --- config -----------------------------------------------------------------
# Stage A trains on PatternReframe alone; stage B continues on the target set.
# Both are 10-way multilabel over the same ml_* columns in the same order, which
# is what lets stage B KEEP stage A's classifier head instead of reinitialising it.
PR_SPLITS = "data/splits_pr_only"        # train = PatternReframe, val/test = target
TARGET_SPLITS = "data/splits_stage2"     # Annotated distorted-only

TAG_A = "mrb-prA"                        # stage A identity
TAG_B = "mrb-prA-annB"                   # stage B identity

SEQ_A_OUT = "results_seq_stageA"
SEQ_B_OUT = "results_seq_stageB"

# Outside the repo clone: cell 1 runs `rm -rf empowerlens`, which would otherwise
# delete stage A's checkpoints and leave stage B with nothing to initialise from.
_ROOT = "/kaggle/working" if Path("/kaggle/working").is_dir() else "."
CK_A = f"{_ROOT}/ckpt_seqA"
CK_B = f"{_ROOT}/ckpt_seqB"

for _d in (SEQ_A_OUT, SEQ_B_OUT):
    Path(_d).mkdir(parents=True, exist_ok=True)
for _d in (CK_A, CK_B):
    Path(_d).mkdir(parents=True, exist_ok=True)


def stage_a_ckpt(seed):
    return f"{CK_A}/multilabel_{TAG_A}_{seed}"


def run_stage(init_model, tag, splits_dir, out_dir, seed, ckpt_root,
              extra_flags="", eval_flags="--allow-distorted-only", do_eval=True):
    """Train one stage, evaluate it, report, sync. Returns the checkpoint path.

    Skips if the eval JSON already exists, so re-running after a restart does not
    retrain a stage that finished.

    `--tag` is not optional here. --model doubles as the run's identity, and for
    stage B that is a local checkpoint path, which would yield
    eval_multilabel_mental-roberta-base_42_multilabel_42.json — a name nothing
    looks for. --tag sets the identity; meta.json records init_from so the
    provenance chain back to stage A survives.

    eval_flags defaults to --allow-distorted-only because both stages train and
    evaluate on distorted-only splits. Those are ISOLATED numbers, comparable to
    Stage 2 isolated (macro_f1 0.277) and never to a cascade result.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_root).mkdir(parents=True, exist_ok=True)
    ckpt = f"{ckpt_root}/multilabel_{tag}_{seed}"
    ev = Path(out_dir) / f"eval_{tag}_multilabel_{seed}.json"

    if ev.exists():
        print(f"  [seed {seed}] {tag}: already completed — skipping")
    else:
        rc = sh(f"python -m src.train_transformer --task multilabel "
                f"--model {init_model} --tag {tag} --seed {seed} --device auto "
                f"--splits {splits_dir} --out {ckpt_root} {extra_flags}",
                timeout=TRAIN_TIMEOUT)
        if rc != 0:
            print(f"  [seed {seed}] {tag}: TRAIN FAILED (rc={rc})")
            return ckpt
        if do_eval:
            print("Waiting 15 seconds for GPU VRAM to flush...")
            time.sleep(15)
            sh(f"python -m src.evaluate --checkpoint {ckpt} --reference "
               f"--splits {splits_dir} --out {out_dir} {eval_flags}",
               timeout=EVAL_TIMEOUT)
        # Optimizer/scheduler state evaluate.py never reads, and the main disk hog.
        # Stage A's model weights are NOT touched — stage B still needs them.
        sh(f"rm -rf {ckpt}/checkpoint-*")

    if ev.exists():
        m = json.loads(ev.read_text())["splits"]["test"]["metrics"]
        print(f"  [seed {seed}] {tag}: test macro_f1={m['macro_f1']:.3f}  "
              f"weighted_f1={m['weighted_f1']:.3f}")
    sync(out_dir)
    return ckpt


print(f"[seq-bootstrap] stageA->{SEQ_A_OUT} stageB->{SEQ_B_OUT}  ckpts={CK_A}, {CK_B}")
