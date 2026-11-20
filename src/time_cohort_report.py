"""
D27 rows E-01 / E-02: time a 50-reflection cohort report end to end.

The report pipeline is what the app runs when a trainer opens a cohort report, on the D25 models
(the D12 winner + D13 thresholds + D03 selector, folder with MODEL_CARD.json):
  1. binary decision   (P(distorted) >= D13 threshold)
  2. distortion types  (per-type val thresholds, at most 2, only for flagged texts)
  3. sentence selection (D03 selector on each sentence of flagged texts)
  4. aggregation       (count per type, share flagged, one example sentence per type shown,
                        types with fewer texts than the cohort threshold suppressed)
Model loading is timed separately and included in the total.

Input: the first N rows of data/splits/val.csv (never test). Output: JSON with times and hardware.

    python -m src.time_cohort_report --models <app_models or models/d12> [--n 50] [--device auto]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DISTORTIONS
from src.span_check import split_sentences
from src.train_transformer import TEXT_COL, encode_texts, load_split, resolve_device

COHORT_THRESHOLD = 5  # the demo app's default report threshold


def ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
        m = MS(); m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.total / 2**30, 1)
    try:
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 2**20, 1)
    except OSError:
        return None


def cpu_name():
    if os.name != "nt":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Time a cohort report (D27 E-01 / E-02)")
    ap.add_argument("--models", required=True, help="folder holding MODEL_CARD.json from the D25 run")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="optional JSON path")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    from src.evaluate import predict_logits, predictions_from_logits

    device = resolve_device(args.device)
    models = Path(args.models)
    texts = load_split("data/splits", "val")[TEXT_COL].astype(str).head(args.n).reset_index(drop=True)

    t0 = time.perf_counter()
    card = json.loads((models / "MODEL_CARD.json").read_text(encoding="utf-8"))
    b_dir, m_dir = models / card["binary"]["checkpoint"], models / card["multilabel"]["checkpoint"]
    b_meta = json.loads((b_dir / "meta.json").read_text(encoding="utf-8"))
    m_meta = json.loads((m_dir / "meta.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(b_dir))
    binary = AutoModelForSequenceClassification.from_pretrained(str(b_dir)).to(device).eval()
    multilabel = AutoModelForSequenceClassification.from_pretrained(str(m_dir)).to(device).eval()
    selector = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(str(b_dir), num_labels=1))
    selector.load_state_dict(torch.load(models / card["selector"]["checkpoint"], map_location="cpu"))
    selector.to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    with torch.no_grad():
        # 1. binary decision
        enc, _ = encode_texts(texts, tok, b_meta["max_length"], b_meta["truncation"], b_meta.get("head_keep", 128))
        flagged = predictions_from_logits(predict_logits(binary, enc, tok.pad_token_id, device), "binary",
                                          binary_threshold=b_meta["binary_threshold"]).astype(bool)
        # 2. distortion types
        enc, _ = encode_texts(texts, tok, m_meta["max_length"], m_meta["truncation"], m_meta.get("head_keep", 128))
        ml_logits = predict_logits(multilabel, enc, tok.pad_token_id, device)
        ml_probs = 1 / (1 + np.exp(-ml_logits))
        ml_pred = predictions_from_logits(ml_logits, "multilabel", m_meta["thresholds"],
                                          max_labels=card["multilabel"].get("max_labels", 2))
        # 3. sentence selection on flagged texts
        sentences = [(i, texts[i][s:e]) for i in np.flatnonzero(flagged) for s, e in split_sentences(texts[i])]
        sel_scores = []
        for k in range(0, len(sentences), 64):
            b = tok([s for _, s in sentences[k:k + 64]], truncation=True, max_length=card["selector"]["max_length"],
                    padding=True, return_tensors="pt").to(device)
            sel_scores += torch.sigmoid(selector(**b).logits.squeeze(-1).float()).cpu().tolist()
    # 4. aggregation
    counts, best = Counter(), {}
    sel_by_text = {}
    for (i, sent), sc in zip(sentences, sel_scores):
        if sc >= card["selector"]["threshold"]:
            sel_by_text.setdefault(i, []).append((sc, sent))
    for i in np.flatnonzero(flagged):
        for j in np.flatnonzero(ml_pred[i]):
            counts[DISTORTIONS[j]] += 1
            for sc, sent in sel_by_text.get(i, []):
                key = (ml_probs[i, j] * sc, sent)
                best[DISTORTIONS[j]] = max(best.get(DISTORTIONS[j], key), key)
    shown = {k: v for k, v in counts.items() if v >= COHORT_THRESHOLD}
    report = {"submissions": len(texts), "share_flagged": round(float(flagged.mean()), 3),
              "distribution": dict(sorted(shown.items(), key=lambda kv: -kv[1])),
              "types_suppressed": sum(1 for v in counts.values() if 0 < v < COHORT_THRESHOLD),
              "examples": {k: best[k][1] for k in shown if k in best}}
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    out = {
        "n_reflections": len(texts), "source": f"data/splits/val.csv first {len(texts)} rows",
        "model_version": card.get("model_version"), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cpu": cpu_name(), "ram_gb": ram_gb(), "torch": torch.__version__, "os": platform.platform(),
        "seconds_load_models": round(t1 - t0, 2), "seconds_report": round(t2 - t1, 2),
        "seconds_total": round(t2 - t0, 2), "target_seconds_gpu": 300,
        "n_sentences_scored": len(sentences), "report": report,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
