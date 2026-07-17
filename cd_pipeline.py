"""
Replication of the methodology in:
Shreevastava & Foltz (2021), "Detecting Cognitive Distortions from
Patient-Therapist Interactions"

IMPORTANT — READ BEFORE INTERPRETING RESULTS
==============================================
Psych_data.csv (the "Therapist Q&A" style dataset) contains only:
    Question (patient text), Answer (therapist text), Therapist, time
It does NOT contain the 3,000 hand-annotated cognitive-distortion labels
used in the paper (11 classes: 'No distortion' + 10 distortion types).

Per instructions, this script therefore builds the FULL pipeline end-to-end
(feature extraction x classifiers, binary + multi-class) but trains on
SYNTHETIC / PLACEHOLDER labels that are randomly generated to mimic the
paper's reported label distribution (39.2% No-Distortion, remaining
60.8% split ~evenly across 10 distortion types). This produces WORKING,
RUNNABLE CODE that mirrors the paper's design exactly — it does NOT
produce scientifically meaningful results, since the labels carry no
real signal from the text. Swap in real annotations (see
`load_real_labels()` stub) to get genuine results.

Offline-environment substitutions (no access to huggingface.co, GloVe
mirrors, or NLTK download servers from this sandbox):
  - SIF semantic embeddings : Word2Vec (Skip-gram) trained ON THIS CORPUS
                              (in place of pretrained GloVe), then SIF-weighted
                              and averaged, per Arora et al. (2016).
  - S-BERT stand-in         : Doc2Vec (Distributed Memory) trained on this
                              corpus — a self-trained *sequential* semantic
                              embedding, used as the closest offline
                              approximation to a pretrained transformer.
  - LIWC                    : A compact hand-built lexicon covering the
                              same category types the paper highlights
                              (pronouns, negative emotion, future focus,
                              feel/perception words, negations, etc.) since
                              the licensed LIWC dictionary isn't available.
  - POS tag embeddings      : spaCy (en_core_web_sm, installed from a GitHub
                              release wheel) for real POS tagging, then
                              Word2Vec (Skip-gram) trained on POS-tag
                              sequences, exactly as the paper describes.
"""

import re
import warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

DISTORTION_TYPES = [
    "Emotional Reasoning", "Overgeneralization", "Mental Filter",
    "Should Statements", "All-or-Nothing", "Mind Reading",
    "Fortune Telling", "Magnification", "Personalization", "Labeling",
]

# ---------------------------------------------------------------------------
# 1. LOAD DATA  (patient input only, as in the paper section 2.1)
# ---------------------------------------------------------------------------

def load_data(path="/mnt/user-data/uploads/Psych_data.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Question"]).reset_index(drop=True)
    df["text"] = df["Question"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. SYNTHETIC / PLACEHOLDER LABELS
#    (Replace this with load_real_labels() once annotations exist)
# ---------------------------------------------------------------------------

def make_synthetic_labels(n, seed=RANDOM_STATE):
    """
    Randomly assigns labels matching the paper's reported distribution:
    39.2% 'No Distortion', 60.8% split evenly across the 10 distortion types.
    THESE LABELS CARRY NO REAL SIGNAL — placeholders only, per user request.
    """
    r = np.random.default_rng(seed)
    p_none = 0.392
    p_each = (1 - p_none) / 10
    classes = ["No Distortion"] + DISTORTION_TYPES
    probs = [p_none] + [p_each] * 10
    labels = r.choice(classes, size=n, p=probs)
    binary = np.where(labels == "No Distortion", "Non-Distorted", "Distorted")
    return pd.Series(labels, name="distortion_type"), pd.Series(binary, name="binary_label")


def load_real_labels(path=None):
    """Stub: point this at a CSV with a 'label' column of real annotations."""
    raise NotImplementedError("No real annotation file was provided.")


# ---------------------------------------------------------------------------
# 3. FEATURE EXTRACTORS
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]


# --- 3a. SIF over self-trained Word2Vec (GloVe substitute) -----------------

def train_word2vec(token_lists, size=100, window=5, min_count=2, seed=RANDOM_STATE):
    from gensim.models import Word2Vec
    model = Word2Vec(
        sentences=token_lists, vector_size=size, window=window,
        min_count=min_count, sg=1, workers=1, seed=seed, epochs=10,
    )
    return model


def sif_embeddings(token_lists, w2v_model, a=1e-3):
    """Smooth Inverse Frequency sentence embeddings (Arora et al., 2016)."""
    vocab_counts = Counter(w for toks in token_lists for w in toks)
    total = sum(vocab_counts.values())
    dim = w2v_model.vector_size
    vecs = np.zeros((len(token_lists), dim))
    for i, toks in enumerate(token_lists):
        weighted = []
        for w in toks:
            if w in w2v_model.wv:
                pw = vocab_counts[w] / total
                weight = a / (a + pw)
                weighted.append(weight * w2v_model.wv[w])
        vecs[i] = np.mean(weighted, axis=0) if weighted else np.zeros(dim)
    # remove first principal component (standard SIF step)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=1, random_state=RANDOM_STATE)
    svd.fit(vecs)
    pc = svd.components_
    vecs = vecs - vecs.dot(pc.T).dot(pc)
    return vecs


# --- 3b. Doc2Vec sequential embeddings (S-BERT substitute) -----------------

def doc2vec_embeddings(token_lists, size=100, seed=RANDOM_STATE):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    docs = [TaggedDocument(toks, [i]) for i, toks in enumerate(token_lists)]
    model = Doc2Vec(
        docs, vector_size=size, window=5, min_count=2, dm=1,
        workers=1, seed=seed, epochs=20,
    )
    return np.array([model.dv[i] for i in range(len(token_lists))])


# --- 3c. LIWC-style lexicon features ---------------------------------------

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


# --- 3d. POS tag embeddings via spaCy + Skip-gram --------------------------

def pos_tag_sequences(texts, nlp, batch_size=64):
    seqs = []
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["ner", "lemmatizer"]):
        seqs.append([tok.pos_ for tok in doc if not tok.is_space])
    return seqs


def pos_embeddings(pos_seqs, size=30, seed=RANDOM_STATE):
    from gensim.models import Word2Vec
    model = Word2Vec(
        sentences=pos_seqs, vector_size=size, window=3, min_count=1,
        sg=1, workers=1, seed=seed, epochs=10,
    )
    dim = model.vector_size
    max_len = 40  # pad/truncate for a fixed-size representation
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


def evaluate_feature_set(X, y, average="weighted"):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import f1_score

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}
    for name, clf in get_classifiers().items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = f1_score(y_test, preds, average=average)
    return results


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    df = load_data()
    # subsample for tractable runtime of the full 4-feature x 5-classifier grid
    N = min(1200, len(df))
    df = df.sample(n=N, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Using {len(df)} patient entries (subsampled for runtime).")

    print("Generating SYNTHETIC placeholder labels (see module docstring)...")
    dist_labels, bin_labels = make_synthetic_labels(len(df))
    df["distortion_type"] = dist_labels
    df["binary_label"] = bin_labels

    texts = df["text"].tolist()
    token_lists = [tokenize(t) for t in texts]

    print("Training Word2Vec (SIF substrate)...")
    w2v = train_word2vec(token_lists)
    X_sif = sif_embeddings(token_lists, w2v)

    print("Training Doc2Vec (S-BERT substitute)...")
    X_d2v = doc2vec_embeddings(token_lists)

    print("Computing LIWC-style lexicon features...")
    X_liwc = liwc_features(texts)

    print("Running spaCy POS tagging + Skip-gram POS embeddings...")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    pos_seqs = pos_tag_sequences(texts, nlp)
    X_pos = pos_embeddings(pos_seqs)

    print("Building hybrid BERT+LIWC feature set...")
    X_hybrid = np.hstack([X_d2v, X_liwc])

    feature_sets = {
        "SIF": X_sif,
        "S-BERT(sub)": X_d2v,
        "LIWC": X_liwc,
        "POS": X_pos,
        "S-BERT+LIWC": X_hybrid,
    }

    print("\n=== BINARY CLASSIFICATION (Distorted vs Non-Distorted) ===")
    binary_table = {}
    for fname, X in feature_sets.items():
        print(f"  evaluating {fname}...")
        binary_table[fname] = evaluate_feature_set(X, df["binary_label"])
    binary_df = pd.DataFrame(binary_table).round(2)
    print(binary_df)

    print("\n=== MULTI-CLASS CLASSIFICATION (Type of Distortion) ===")
    multi_table = {}
    for fname, X in feature_sets.items():
        print(f"  evaluating {fname}...")
        multi_table[fname] = evaluate_feature_set(X, df["distortion_type"])
    multi_df = pd.DataFrame(multi_table).round(2)
    print(multi_df)

    binary_df.to_csv("/home/claude/binary_f1_results.csv")
    multi_df.to_csv("/home/claude/multiclass_f1_results.csv")
    df[["text", "binary_label", "distortion_type"]].to_csv(
        "/home/claude/labeled_data_sample.csv", index=False
    )
    print("\nDone. Results saved to binary_f1_results.csv / multiclass_f1_results.csv")
    return binary_df, multi_df


if __name__ == "__main__":
    main()
