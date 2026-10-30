"""Train + compare 3 TF-IDF models, save best model and report.
Run:  python train.py   (after python generate_data.py)
"""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from common import normalize

SEED = 42
ROOT = Path(__file__).parent
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"


def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                                 sublinear_tf=True, min_df=2)),
    ])


CANDIDATES = {
    "LogReg": lambda: LogisticRegression(C=20, max_iter=2000),
    # LinearSVC has no probabilities, so it is wrapped with probability calibration
    "LinearSVM": lambda: CalibratedClassifierCV(LinearSVC(C=1.0), cv=3),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=200,
                                 random_state=SEED),
}


def group_split(df, frac=0.2):
    """Split by template_group so variations of one template never span train and test."""
    rng, test_groups = np.random.RandomState(SEED), set()
    for _, g in df.groupby("intent"):
        groups = sorted(g.template_group.unique())
        rng.shuffle(groups)
        test_groups.update(groups[:max(1, round(len(groups) * frac))])
    mask = df.template_group.isin(test_groups)
    train, test = df[~mask], df[mask]
    seen = set(train.utterance.map(normalize))        # remove normalized duplicates across splits
    return train, test[~test.utterance.map(normalize).isin(seen)]


def latency_ms(pipe, texts, n=200):
    for t in texts[:10]:
        pipe.predict_proba([t])                        # warm-up
    times = []
    for i in range(n):
        s = time.perf_counter()
        pipe.predict_proba([texts[i % len(texts)]])
        times.append((time.perf_counter() - s) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "commands.csv")
    hw = pd.read_csv(DATA / "handwritten_test.csv")
    train, test = group_split(df)
    train.to_csv(DATA / "train.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)
    print(f"train={len(train)} test={len(test)} handwritten={len(hw)}")

    rows, per_intent, fitted = [], {}, {}
    for name, make in CANDIDATES.items():
        pipe = Pipeline([("tfidf", features()), ("clf", make())])
        t0 = time.perf_counter()
        pipe.fit(train.utterance, train.intent)
        fit_s = time.perf_counter() - t0
        pred = pipe.predict(test.utterance)
        hw_pred = pipe.predict(hw.utterance)
        lat_med, lat_p95 = latency_ms(pipe, list(test.utterance))
        path = MODELS / f"{name}.joblib"
        joblib.dump(pipe, path, compress=3)
        rep = classification_report(test.intent, pred, output_dict=True, zero_division=0)
        per_intent[name] = {k: round(v["f1-score"], 3) for k, v in rep.items()
                            if k not in ("accuracy", "macro avg", "weighted avg")}
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(test.intent, pred), 4),
            "macro_f1": round(f1_score(test.intent, pred, average="macro"), 4),
            "handwritten_acc": round(accuracy_score(hw.intent, hw_pred), 4),
            "handwritten_macro_f1": round(f1_score(hw.intent, hw_pred, average="macro"), 4),
            "latency_ms_median": round(lat_med, 2),
            "latency_ms_p95": round(lat_p95, 2),
            "size_kb": round(path.stat().st_size / 1024, 1),
            "train_s": round(fit_s, 1),
        })
        fitted[name] = (pipe, hw_pred)
        print(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "model_comparison.csv", index=False)
    pi = pd.DataFrame(per_intent)
    pi.index.name = "intent"
    pi.reset_index().to_csv(REPORTS / "per_intent_f1.csv", index=False)

    # Selection: within 0.01 macro-F1 of the top model, pick the lowest latency
    top = res.macro_f1.max()
    pool = res[res.macro_f1 >= top - 0.01].sort_values(["latency_ms_median", "size_kb"])
    best = pool.iloc[0]
    shutil.copy(MODELS / f"{best.model}.joblib", MODELS / "best_model.joblib")
    (MODELS / "meta.json").write_text(json.dumps(best.to_dict(), indent=2, default=str))

    hw_pred = fitted[best.model][1]
    miss = hw.assign(predicted=hw_pred)
    miss = miss[miss.intent != miss.predicted].rename(columns={"intent": "expected"})
    top_row = res.loc[res.macro_f1.idxmax()]
    report = f"""# Model comparison report

All numbers below were measured by `train.py` on this machine (CPU, single-utterance inference).
Test split = held-out template groups ({len(test)} utterances); handwritten set = {len(hw)} unseen phrases.

## Overall
{md_table(res)}

## Per-intent F1 (held-out template groups)
{md_table(pi.reset_index())}

## Selection
**Selected: {best.model}.** Rule: keep models within 0.01 macro-F1 of the top one
({top_row.model}, {top_row.macro_f1}), then pick the lowest median latency.
{best.model}: macro-F1 {best.macro_f1}, accuracy {best.accuracy}, median latency {best.latency_ms_median} ms,
size {best.size_kb} KB.

## Handwritten-set misses for {best.model} (raw classifier output, before safeguards)
{md_table(miss) if len(miss) else "None."}

## Caveats
- Data is template-generated, so the held-out score is optimistic; the handwritten set is the better
  indicator of real behaviour. Both are small.
- Confidence is calibrated `predict_proba` (LinearSVM is wrapped in CalibratedClassifierCV).
- Validate with real Whisper transcripts and driver audio before any real use.
"""
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    print(f"\nBest model: {best.model}  -> models/best_model.joblib\nReport: reports/report.md")


if __name__ == "__main__":
    main()
