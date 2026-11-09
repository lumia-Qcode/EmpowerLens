"""
D03 - sentence-level span selector: scores every sentence of a reflection for
"does this sentence carry the distortion?" and selects those above a threshold.

RUN (one seed; repeat for 42, 1337, 2024):
    python -m src.train_span_selector --seed 42

Input : results/results_sentence_label/sentence_labels_{train,val,test}.csv (T12; has `split`
        column that follows data/splits, so all sentences of a text stay in one split).
        Only distorted rows with a located span are in there (13 unlocated rows excluded).
Output: results/results_span_selector/
          config.json                          settings (written once)
          eval_val_<seed>.json  eval_test_<seed>.json
          predictions_val_<seed>.csv  predictions_test_<seed>.csv
Rules : the threshold is chosen on VAL only (max sentence F1) and applied unchanged to test;
        test is scored once, after training. Checkpoints go to checkpoints/ (never committed).
Model : one encoder sees ONE sentence (no context), max length 128; BCE with pos_weight
        for the ~16% positive rate.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_DIR = Path("results/results_sentence_label")
OUT_DIR = Path("results/results_span_selector")
ID_COL = "Id_Number"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows torch CPU OpenMP crash


def load_part(part):
    df = pd.read_csv(LABELS_DIR / f"sentence_labels_{part}.csv", encoding="utf-8")
    return df.rename(columns={"id": ID_COL}).reset_index(drop=True)


def prf(gold, sel):
    gold, sel = np.asarray(gold, bool), np.asarray(sel, bool)
    tp = int((gold & sel).sum())
    p = tp / sel.sum() if sel.sum() else 0.0
    r = tp / gold.sum() if gold.sum() else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_threshold(gold, score):
    """Threshold maximising sentence F1 (call on VAL only)."""
    grid = np.unique(np.round(score, 4))
    return float(max(grid, key=lambda t: (prf(gold, score >= t)[2], -t)))


def score_part(df, score, thr):
    """Sentence-level P/R/F1 + row hit rate (>=1 selected sentence is gold)."""
    sel = score >= thr
    p, r, f = prf(df["label"].values, sel)
    hit = (df.assign(h=(df["label"].values == 1) & sel).groupby(ID_COL)["h"].max()).mean()
    return {"precision": p, "recall": r, "f1": f, "row_hit_rate": float(hit),
            "n_rows": int(df[ID_COL].nunique()), "n_sentences": int(len(df)),
            "n_gold_sentences": int(df["label"].sum()), "n_selected": int(sel.sum()),
            "threshold": thr}


def save_part(df, score, thr, part, seed, extra):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = score_part(df, score, thr)
    res.update(extra, split=part, seed=seed)
    (OUT_DIR / f"eval_{part}_{seed}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame({ID_COL: df[ID_COL], "sent_idx": df["sent_idx"], "sentence": df["sentence"],
                  "score": np.round(score, 5), "selected": np.where(score >= thr, "yes", "no"),
                  "gold": np.where(df["label"] == 1, "yes", "no")}
                 ).to_csv(OUT_DIR / f"predictions_{part}_{seed}.csv", index=False, encoding="utf-8")
    print(part, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="mental/mental-roberta-base")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true", help="200 train sentences, 1 epoch")
    ap.add_argument("--ckpt-dir", default="checkpoints/span_selector")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = load_part("train"), load_part("val"), load_part("test")
    if a.smoke:
        tr, va, te, a.epochs = tr.head(200), va.head(100), te.head(100), 1

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)

    def encode(texts):
        return tok(list(texts), truncation=True, max_length=a.max_length, padding=True, return_tensors="pt")

    def predict(df):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(df), 64):
                b = encode(df["sentence"].values[i:i + 64]).to(dev)
                out.append(torch.sigmoid(model(**b).logits.squeeze(-1)).cpu().numpy())
        return np.concatenate(out)

    y = tr["label"].values
    pos_w = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=torch.float, device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * ((len(tr) + a.batch_size - 1) // a.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    best_f1, best_state, best_thr, best_ep = -1, None, 0.5, 0
    for ep in range(a.epochs):
        model.train()
        order = np.random.RandomState(a.seed + ep).permutation(len(tr))
        for i in range(0, len(order), a.batch_size):
            idx = order[i:i + a.batch_size]
            b = encode(tr["sentence"].values[idx]).to(dev)
            lab = torch.tensor(tr["label"].values[idx], dtype=torch.float, device=dev)
            loss = loss_fn(model(**b).logits.squeeze(-1), lab)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        s = predict(va); thr = best_threshold(va["label"].values, s); f1 = prf(va["label"].values, s >= thr)[2]
        print(f"epoch {ep + 1}: val sentence F1 {f1:.4f} @ thr {thr:.3f}")
        if f1 > best_f1:   # epoch chosen on val
            best_f1, best_thr, best_ep = f1, thr, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    Path(a.ckpt_dir, str(a.seed)).mkdir(parents=True, exist_ok=True)
    torch.save(best_state, Path(a.ckpt_dir, str(a.seed), "model.pt"))   # checkpoints/ is gitignored

    extra = {"model": a.model, "best_epoch": best_ep}
    save_part(va, predict(va), best_thr, "val", a.seed, extra)
    save_part(te, predict(te), best_thr, "test", a.seed, extra)   # test scored once, val threshold

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {"model": a.model, "lr": a.lr, "epochs": a.epochs, "batch_size": a.batch_size,
           "max_length": a.max_length, "weight_decay": a.weight_decay, "loss": "BCE pos_weight=neg/pos",
           "input": "single sentence, no context", "threshold": "per seed, max val sentence F1 (see eval_val_<seed>.json)",
           "seeds": [42, 1337, 2024], "labels": "results/results_sentence_label (T12; 13 unlocated rows excluded)"}
    if not a.smoke:
        (OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
