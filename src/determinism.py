"""
Make training reproducible run-to-run, not just seed-to-seed.

THE PROBLEM
-----------
Setting a seed makes the *random* parts repeatable — weight initialisation,
shuffling, dropout. It does not make the *arithmetic* repeatable.

A GPU adds thousands of numbers at once and accumulates them in whatever order
its threads happen to finish. Floating-point addition is order-dependent: the
same 100,000 floats summed forwards, backwards and shuffled give

    -90.825134    -90.825070    -90.825070

A single step's discrepancy is ~1e-5. Over ~500 steps per epoch, each step
feeding the next, it compounds into a measurably different model.

Measured cost in this project (docs/SUPERVISOR_EXPERIMENTS.md): Experiments 6
and 7 were the same configuration — same backbone, splits, loss, seeds, only the
output directory differed — and disagreed by **0.047 macro-F1 at seed 42**, while
the reported seed-to-seed SD was **0.007**. Run-to-run noise was seven times the
error bar being published, which makes any difference under ~0.05 unreadable.

WHAT THIS FIXES, AND WHAT IT DOES NOT
-------------------------------------
``enable_determinism()`` pins the algorithm choice so the same seed produces the
same arithmetic. Verified locally: two runs gave bit-identical loss and gradients.

It does **not** make different seeds agree — that is the variance you actually
want to measure and report.

``warn_only=True`` is deliberate. Some ops have no deterministic CUDA kernel; with
``warn_only=False`` those raise and the run dies, which on a timed Kaggle session
costs hours. With ``warn_only=True`` they warn and fall back, so the run finishes
— but a warning means that op is still non-deterministic. **This is why the
verification run is not optional**: it is the only thing that turns "determinism
requested" into "determinism demonstrated".

CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialises, so call this at the
very top of a script, before anything imports torch.

Usage
-----
    from src.determinism import enable_determinism
    enable_determinism()          # before torch-dependent training code

    python -m src.determinism --check      # prove it on this machine
"""

from __future__ import annotations

import os
import warnings

# cuBLAS needs a fixed workspace size for deterministic GEMMs on CUDA >= 10.2.
# Harmless on CPU. Must precede CUDA initialisation, hence module import time is
# too late if torch has already been imported and used.
_CUBLAS = ":4096:8"


def enable_determinism(warn_only: bool = True, seed: int | None = None) -> dict:
    """Pin every source of run-to-run nondeterminism we can reach.

    Returns a dict describing what was set, suitable for recording in a run's
    meta.json — a result is only as reproducible as the settings it was made
    under, so those settings belong with the result.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS)
    os.environ.setdefault("PYTHONHASHSEED", "0")

    import torch

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    # cuDNN benchmarking times several algorithms and keeps the fastest ON THAT
    # RUN, so the winner depends on machine load. That alone breaks run-to-run
    # reproducibility even with everything else pinned.
    torch.backends.cudnn.benchmark = False

    if seed is not None:
        import random

        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return {
        "deterministic": True,
        "warn_only": warn_only,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
        "seed": seed,
    }


def _check() -> int:
    """Train a tiny model twice at the same seed and compare bit-for-bit."""
    import sys

    info = enable_determinism()
    import torch
    from transformers import AutoModelForSequenceClassification, set_seed

    print("settings:", {k: v for k, v in info.items() if k != "torch_version"})
    print(f"torch {info['torch_version']}  cuda={torch.cuda.is_available()}\n")

    def one_run():
        set_seed(42)
        m = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=10,
            problem_type="multi_label_classification").float()
        if torch.cuda.is_available():
            m = m.cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
        g = torch.Generator().manual_seed(0)
        losses = []
        for _ in range(5):                     # a few real optimizer steps
            x = torch.randint(0, 1000, (4, 32), generator=g)
            y = (torch.rand(4, 10, generator=g) < 0.2).float()
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            out = m(input_ids=x, labels=y)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(out.loss.detach().float().item())
        return losses

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = one_run(), one_run()

    print("run 1:", [f"{v:.10f}" for v in a])
    print("run 2:", [f"{v:.10f}" for v in b])
    gap = max(abs(x - y) for x, y in zip(a, b))
    print(f"\nmax difference: {gap:.3e}")
    if a == b:
        print("PASS - bit-identical. Same seed now reproduces exactly.")
        return 0
    print("FAIL - the runs differ. Error bars from this machine are not "
          "trustworthy;\ninvestigate before quoting any +/- SD.")
    return 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true",
                    help="train a tiny model twice at one seed and compare")
    args = ap.parse_args(argv)
    if args.check:
        return _check()
    print(enable_determinism())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
