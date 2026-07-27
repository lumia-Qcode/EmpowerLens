"""
Replication of the methodology in:
Shreevastava & Foltz (2021), "Detecting Cognitive Distortions from
Patient-Therapist Interactions"

FIXES APPLIED (vs. original cd_pipeline.py)
==============================================
1. TRANSDUCTIVE LEAKAGE FIXED
   Previously every embedding method (Word2Vec/SIF, Doc2Vec, POS-Word2Vec,
   SIF's SVD "common component removal") was fit on ALL rows (train+val+test)
   before the classifier's own train_test_split. This is a real leak:
     - Doc2Vec (dm=1) learns a parameter vector per document during
       training, so test-set text was literally optimized into the
       embedding space it was later "evaluated" on.
     - Word2Vec vocab/frequencies and SIF's `a/(a+pw)` weighting were
       computed from corpus stats that included val+test text.
     - The SIF TruncatedSVD (common-component removal) was fit on the
       full matrix, so the "noise direction" subtracted was shaped by
       held-out documents too.
     - POS-tag Word2Vec had the same full-corpus-fit problem.
   FIX: every embedding model below is fit ONLY on the training split's
   text. Val/test vectors are produced by *inference* against the
   already-fitted model (Word2Vec: average of known word vectors;
   Doc2Vec: model.infer_vector(); SIF: apply train-fit vocab weights +
   project out the train-fit principal component, no refitting; POS
   Word2Vec: same pattern as word-level Word2Vec).

2. PRE-COMPUTED SPLITS NOW USED
   Previously `load_data()` read a nonexistent "Annotated_data.csv" and
   `evaluate_feature_set()` did its own fresh 80/20 `train_test_split`
   *every time it was called* (once per feature set x per task), so the
   test set silently differed between SIF/Doc2Vec/LIWC/POS/hybrid runs
   and between binary vs. multiclass runs, and val.csv was never touched.
   FIX: this script loads train.csv / val.csv / test.csv directly (the
   iterative-stratification split described in split_manifest.json) and
   reuses the exact same three splits for every feature set and every
   task, so results are reproducible and comparable. val.csv is used for
   a val-set F1 as well as test.csv, matching the manifest's 80/10/10
   split.

Everything else (LIWC lexicon features, spaCy POS tagging, classifier
choices, offline substitutions for GloVe/S-BERT) is unchanged from the
original script.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from typing import Literal

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

DISTORTION_TYPES = [
    "Emotional Reasoning", "Overgeneralization", "Mental Filter",
    "Should Statements", "All-or-Nothing", "Mind Reading",
    "Fortune Telling", "Magnification", "Personalization", "Labeling",
]

LABEL_NORMALIZATION = {
    "no distortion": "No Distortion",
    "emotional reasoning": "Emotional Reasoning",
    "overgeneralization": "Overgeneralization",
    "mental filter": "Mental Filter",
    "should statements": "Should Statements",
    "all-or-nothing thinking": "All-or-Nothing",
    "all or nothing": "All-or-Nothing",
    "mind reading": "Mind Reading",
    "fortune-telling": "Fortune Telling",
    "fortune telling": "Fortune Telling",
    "magnification": "Magnification",
    "personalization": "Personalization",
    "labeling": "Labeling",
}


def _normalize_label(raw):
    key = str(raw).strip().lower()
    return LABEL_NORMALIZATION.get(key, str(raw).strip())


# ---------------------------------------------------------------------------
# 1. LOAD PRE-COMPUTED SPLITS (train.csv / val.csv / test.csv)
# ---------------------------------------------------------------------------

def _resolve_split_dir():
    """
    Looks for train.csv/val.csv/test.csv in a few likely locations so this
    script works whether it's run from the repo root, from data/splits, or
    with the CSVs sitting next to the script. Set DATA_SPLIT_DIR env var
    (or edit `candidates` below) to point at a different location.
    """
    candidates = [
        os.environ.get("DATA_SPLIT_DIR", ""),
        os.path.join("data", "splits"),
        "splits",
        ".",
    ]
    for c in candidates:
        if c and all(
            os.path.isfile(os.path.join(c, f))
            for f in ("train.csv", "val.csv", "test.csv")
        ):
            return c
    raise FileNotFoundError(
        "Could not find train.csv/val.csv/test.csv. Set DATA_SPLIT_DIR "
        "to the folder containing them, e.g. DATA_SPLIT_DIR=data/splits"
    )


def _prep(df):
    df = df.dropna(subset=["Patient Question", "Dominant Distortion"]).reset_index(drop=True)
    df["text"] = df["Patient Question"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    df["distortion_type"] = df["Dominant Distortion"].apply(_normalize_label)
    df["binary_label"] = np.where(
        df["distortion_type"] == "No Distortion", "Non-Distorted", "Distorted"
    )
    return df


def load_splits(split_dir=None):
    """
    Loads the REAL pre-computed, iteratively-stratified train/val/test
    splits (see split_manifest.json) instead of re-splitting from scratch.
    Returns three DataFrames: train_df, val_df, test_df.
    """
    split_dir = split_dir or _resolve_split_dir()
    train_df = _prep(pd.read_csv(os.path.join(split_dir, "train.csv")))
    val_df = _prep(pd.read_csv(os.path.join(split_dir, "val.csv")))
    test_df = _prep(pd.read_csv(os.path.join(split_dir, "test.csv")))
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 2. TOKENIZATION
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# 3. FEATURE EXTRACTORS — each one is FIT ON TRAIN ONLY, then used to
#    TRANSFORM val/test (no refitting, no peeking at held-out text).
# ---------------------------------------------------------------------------

# --- 3a. SIF over self-trained Word2Vec (GloVe substitute) -----------------

def train_word2vec(token_lists, size=100, window=5, min_count=2, seed=RANDOM_STATE):
    from gensim.models import Word2Vec
    model = Word2Vec(
        sentences=token_lists, vector_size=size, window=window,
        min_count=min_count, sg=1, workers=1, seed=seed, epochs=10,
    )
    return model


def fit_sif(train_token_lists, w2v_model, a=1e-3):
    """
    Fits SIF weighting (word-frequency table from TRAIN only) and the
    common-component (TruncatedSVD) direction on TRAIN embeddings only.
    Returns a `transform(token_lists)` closure that can be applied to any
    split (train, val, or test) without refitting anything.
    """
    from sklearn.decomposition import TruncatedSVD

    vocab_counts = Counter(w for toks in train_token_lists for w in toks)
    total = sum(vocab_counts.values())
    dim = w2v_model.vector_size

    def _raw_sif(token_lists):
        vecs = np.zeros((len(token_lists), dim))
        for i, toks in enumerate(token_lists):
            weighted = []
            for w in toks:
                if w in w2v_model.wv:
                    pw = vocab_counts.get(w, 0) / total if total else 0
                    weight = a / (a + pw) if pw > 0 else 1.0
                    weighted.append(weight * w2v_model.wv[w])
            vecs[i] = np.mean(weighted, axis=0) if weighted else np.zeros(dim)
        return vecs

    train_vecs = _raw_sif(train_token_lists)
    svd = TruncatedSVD(n_components=1, random_state=RANDOM_STATE)
    svd.fit(train_vecs)  # fit on TRAIN only
    pc = svd.components_

    def transform(token_lists):
        vecs = _raw_sif(token_lists)
        return vecs - vecs.dot(pc.T).dot(pc)  # project out train-fit component

    return transform


# --- 3b. Doc2Vec sequential embeddings (S-BERT substitute) -----------------

def train_doc2vec(train_token_lists, size=100, seed=RANDOM_STATE):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    docs = [TaggedDocument(toks, [i]) for i, toks in enumerate(train_token_lists)]
    model = Doc2Vec(
        docs, vector_size=size, window=5, min_count=2, dm=1,
        workers=1, seed=seed, epochs=20,
    )
    return model


def doc2vec_transform(token_lists, model, is_train, infer_epochs=20):
    """
    For TRAIN docs (which the model was actually trained on) we can pull
    the learned vector directly via model.dv[i]. For VAL/TEST docs we use
    infer_vector(), which holds the model's learned word vectors fixed and
    only optimizes a *new* doc vector for the unseen text — this does not
    touch/update the trained model, unlike calling train() again.
    """
    if is_train:
        return np.array([model.dv[i] for i in range(len(token_lists))])
    return np.array([
        model.infer_vector(toks, epochs=infer_epochs) for toks in token_lists
    ])


# --- 3c. LIWC-style lexicon features (deterministic, no fitting needed) ----

LIWC_LEXICON = {
    "pronoun_1p": {"i", "me", "my", "mine", "myself"},
    "pronoun_3p": {"he", "him", "his", "she", "her", "hers", "they", "them", "their"},
    "negation": {"not", "no", "never", "cant", "cannot", "wont", "dont", "didnt", "isnt"},
    "negative_emotion": {"sad", "angry", "afraid", "anxious", "worried", "scared",
                          "upset", "hopeless", "worthless", "guilty", "ashamed",
                          "hate", "fear", "depressed", "miserable", "hurt", "lonely"},
    "positive_emotion": {"happy", "good", "love", "hope", "glad", "grateful",
                          "calm", "confident", "proud", "relieved"},
    "feel_perception": {"feel", "feels", "feeling", "felt", "sense", "seem", "seems", "notice"},
    "future_focus": {"will", "going", "gonna", "future", "someday", "eventually", "soon"},
    "certainty_absolutes": {"always", "never", "everyone", "nobody", "everything",
                             "nothing", "completely", "totally", "every", "all"},
    "should_words": {"should", "must", "ought", "supposed", "have"},
    "self_blame": {"fault", "blame", "myself", "mistake", "failure", "wrong"},
}


def liwc_features(texts):
    rows = []
    for text in texts:
        toks = tokenize(text)
        n = max(len(toks), 1)
        counts = Counter(toks)
        row = {cat: sum(counts[w] for w in words) / n
               for cat, words in LIWC_LEXICON.items()}
        row["word_count"] = len(toks)
        row["avg_word_len"] = np.mean([len(t) for t in toks]) if toks else 0
        rows.append(row)
    return pd.DataFrame(rows).values


# --- 3d. POS tag embeddings via spaCy + Skip-gram ---------------------------

def pos_tag_sequences(texts, nlp, batch_size=64):
    seqs = []
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["ner", "lemmatizer"]):
        seqs.append([tok.pos_ for tok in doc if not tok.is_space])
    return seqs


def train_pos_word2vec(train_pos_seqs, size=30, seed=RANDOM_STATE):
    from gensim.models import Word2Vec
    model = Word2Vec(
        sentences=train_pos_seqs, vector_size=size, window=3, min_count=1,
        sg=1, workers=1, seed=seed, epochs=10,
    )
    return model


def pos_embeddings_transform(pos_seqs, model, max_len=40):
    dim = model.vector_size
    vecs = np.zeros((len(pos_seqs), dim))
    for i, seq in enumerate(pos_seqs):
        seq = seq[:max_len]
        word_vecs = [model.wv[t] for t in seq if t in model.wv]
        vecs[i] = np.mean(word_vecs, axis=0) if word_vecs else np.zeros(dim)
    return vecs


# ---------------------------------------------------------------------------
# 4. CLASSIFICATION (mirrors paper section 2.2 / 3, default hyperparameters)
# ---------------------------------------------------------------------------

def get_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    return {
        "Log. Reg.": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "k-NN (k=15)": KNeighborsClassifier(n_neighbors=15),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                              random_state=RANDOM_STATE),
    }


def evaluate_feature_set(X_train, y_train, X_val, y_val, X_test, y_test,
                          average: Literal['micro', 'macro', 'samples', 'weighted', 'binary', None] = "weighted"):
    """
    Uses the SAME fixed train/val/test split for every feature set and
    every task (no fresh random split per call, unlike the original).
    Fits StandardScaler + LabelEncoder on TRAIN only.
    Reports F1 on both val and test.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    # unseen labels in val/test would break transform(); guard just in case
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    val_results, test_results = {}, {}
    for name, clf in get_classifiers().items():
        clf.fit(X_train_s, y_train_enc)
        val_results[name] = f1_score(y_val_enc, clf.predict(X_val_s), average=average)
        test_results[name] = f1_score(y_test_enc, clf.predict(X_test_s), average=average)
    return val_results, test_results


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading pre-computed train/val/test splits...")
    train_df, val_df, test_df = load_splits()
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print("\nTrain label distribution (Dominant Distortion):")
    print(train_df["distortion_type"].value_counts())

    train_tok = [tokenize(t) for t in train_df["text"]]
    val_tok = [tokenize(t) for t in val_df["text"]]
    test_tok = [tokenize(t) for t in test_df["text"]]

    # --- SIF (Word2Vec fit on train only) -----------------------------
    print("Training Word2Vec on TRAIN only (SIF substrate)...")
    w2v = train_word2vec(train_tok)
    sif_transform = fit_sif(train_tok, w2v)
    X_sif_train = sif_transform(train_tok)
    X_sif_val = sif_transform(val_tok)
    X_sif_test = sif_transform(test_tok)

    # --- Doc2Vec (fit on train only, infer for val/test) ----------------
    print("Training Doc2Vec on TRAIN only (S-BERT substitute)...")
    d2v = train_doc2vec(train_tok)
    X_d2v_train = doc2vec_transform(train_tok, d2v, is_train=True)
    X_d2v_val = doc2vec_transform(val_tok, d2v, is_train=False)
    X_d2v_test = doc2vec_transform(test_tok, d2v, is_train=False)

    # --- LIWC (deterministic per-doc counts, no fitting) -----------------
    print("Computing LIWC-style lexicon features...")
    X_liwc_train = liwc_features(train_df["text"])
    X_liwc_val = liwc_features(val_df["text"])
    X_liwc_test = liwc_features(test_df["text"])

    # --- POS embeddings (Word2Vec fit on train POS sequences only) -------
    print("Running spaCy POS tagging + Skip-gram POS embeddings (fit on train only)...")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    pos_train = pos_tag_sequences(train_df["text"], nlp)
    pos_val = pos_tag_sequences(val_df["text"], nlp)
    pos_test = pos_tag_sequences(test_df["text"], nlp)
    pos_w2v = train_pos_word2vec(pos_train)
    X_pos_train = pos_embeddings_transform(pos_train, pos_w2v)
    X_pos_val = pos_embeddings_transform(pos_val, pos_w2v)
    X_pos_test = pos_embeddings_transform(pos_test, pos_w2v)

    # --- Hybrid S-BERT(sub) + LIWC ---------------------------------------
    X_hyb_train = np.hstack([X_d2v_train, X_liwc_train])
    X_hyb_val = np.hstack([X_d2v_val, X_liwc_val])
    X_hyb_test = np.hstack([X_d2v_test, X_liwc_test])

    feature_sets = {
        "SIF": (X_sif_train, X_sif_val, X_sif_test),
        "S-BERT(sub)": (X_d2v_train, X_d2v_val, X_d2v_test),
        "LIWC": (X_liwc_train, X_liwc_val, X_liwc_test),
        "POS": (X_pos_train, X_pos_val, X_pos_test),
        "S-BERT+LIWC": (X_hyb_train, X_hyb_val, X_hyb_test),
    }

    print("\n=== BINARY CLASSIFICATION (Distorted vs Non-Distorted) ===")
    binary_val_table, binary_test_table = {}, {}
    for fname, (Xtr, Xv, Xte) in feature_sets.items():
        print(f"  evaluating {fname}...")
        val_res, test_res = evaluate_feature_set(
            Xtr, train_df["binary_label"], Xv, val_df["binary_label"],
            Xte, test_df["binary_label"],
        )
        binary_val_table[fname] = val_res
        binary_test_table[fname] = test_res
    binary_val_df = pd.DataFrame(binary_val_table).round(2)
    binary_test_df = pd.DataFrame(binary_test_table).round(2)
    print("val F1:\n", binary_val_df)
    print("test F1:\n", binary_test_df)

    print("\n=== MULTI-CLASS CLASSIFICATION (Type of Distortion) ===")
    multi_val_table, multi_test_table = {}, {}
    for fname, (Xtr, Xv, Xte) in feature_sets.items():
        print(f"  evaluating {fname}...")
        val_res, test_res = evaluate_feature_set(
            Xtr, train_df["distortion_type"], Xv, val_df["distortion_type"],
            Xte, test_df["distortion_type"],
        )
        multi_val_table[fname] = val_res
        multi_test_table[fname] = test_res
    multi_val_df = pd.DataFrame(multi_val_table).round(2)
    multi_test_df = pd.DataFrame(multi_test_table).round(2)
    print("val F1:\n", multi_val_df)
    print("test F1:\n", multi_test_df)

    binary_val_df.to_csv("binary_f1_val_results.csv")
    binary_test_df.to_csv("binary_f1_test_results.csv")
    multi_val_df.to_csv("multiclass_f1_val_results.csv")
    multi_test_df.to_csv("multiclass_f1_test_results.csv")
    print("\nDone. Results saved to binary_f1_*_results.csv / multiclass_f1_*_results.csv")
    return binary_val_df, binary_test_df, multi_val_df, multi_test_df


if __name__ == "__main__":
    main()