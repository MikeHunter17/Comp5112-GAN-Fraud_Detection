"""Metrics, threshold selection, classifiers, generator diagnostics."""

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    mcc: float
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    def row(self) -> Dict:
        return asdict(self)


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1. Chosen on validation, then frozen."""
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.where(denom == 0, 1, denom), 0.0)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def score(y: np.ndarray, p: np.ndarray, threshold: float) -> Scores:
    y = np.asarray(y, int)
    pred = (np.asarray(p, float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) > 1
    return Scores(
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        pr_auc=float(average_precision_score(y, p)) if both else np.nan,
        roc_auc=float(roc_auc_score(y, p)) if both else np.nan,
        mcc=float(matthews_corrcoef(y, pred)),
        threshold=float(threshold),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def classifier(name: str, seed: int = 0):
    """No class_weight: that would be a second imbalance treatment, confounding the test."""
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                      n_jobs=-1, random_state=seed)
    raise KeyError(name)


def fit_and_score(name, Xtr, ytr, Xva, yva, Xte, yte, seed=0) -> Tuple[Scores, Scores, float]:
    clf = classifier(name, seed)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    secs = time.perf_counter() - t0

    pva = clf.predict_proba(Xva)[:, 1]
    thr = pick_threshold(yva, pva)
    pte = clf.predict_proba(Xte)[:, 1]
    return score(yva, pva, thr), score(yte, pte, thr), secs


# --------------------------------------------------------------------------
# generator diagnostics
# --------------------------------------------------------------------------

@dataclass
class Diagnostics:
    marginal_similarity: float
    diversity_ratio: float
    duplication_rate: float
    coverage: float

    def verdict(self) -> str:
        bad = []
        if self.diversity_ratio < 0.5:
            bad.append("possible mode collapse")
        if self.duplication_rate > 0.20:
            bad.append("possible memorisation")
        if self.marginal_similarity < 0.70:
            bad.append("marginals mismatched")
        if self.coverage < 0.30:
            bad.append("poor coverage")
        return "; ".join(bad) or "no obvious failure"

    def row(self) -> Dict:
        d = {k: round(v, 4) for k, v in asdict(self).items()}
        d["verdict"] = self.verdict()
        return d


def diagnose(real: pd.DataFrame, synth: pd.DataFrame, numeric: List[str],
             categorical: List[str], cap: int = 1500, seed: int = 0) -> Diagnostics:
    """Compare generated rows against the real fraud they were learned from."""
    rng = np.random.default_rng(seed)
    if synth is None or len(synth) == 0:
        return Diagnostics(np.nan, np.nan, np.nan, np.nan)

    sims = []
    for c in numeric:
        a = pd.to_numeric(real[c], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(synth[c], errors="coerce").dropna().to_numpy()
        if len(a) > 1 and len(b) > 1:
            sims.append(1 - ks_2samp(a, b).statistic)
    for c in categorical:
        pa = real[c].astype(str).value_counts(normalize=True)
        pb = synth[c].astype(str).value_counts(normalize=True)
        lv = set(pa.index) | set(pb.index)
        sims.append(1 - 0.5 * sum(abs(pa.get(l, 0) - pb.get(l, 0)) for l in lv))
    marginal = float(np.mean(sims)) if sims else np.nan

    def enc(df):
        parts = [df[numeric].astype(float).reset_index(drop=True)] if numeric else []
        for c in categorical:
            codes = df[c].astype(str).reset_index(drop=True)
            lv = sorted(real[c].astype(str).unique())
            parts.append(pd.DataFrame({f"{c}={v}": (codes == v).astype(float) for v in lv}))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(np.zeros((len(df), 1)))

    R = enc(real)
    S = enc(synth).reindex(columns=R.columns, fill_value=0.0)
    sc = StandardScaler().fit(R.to_numpy())
    Rz, Sz = sc.transform(R.to_numpy()), sc.transform(S.to_numpy())

    sub = lambda M: M if len(M) <= cap else M[rng.choice(len(M), cap, replace=False)]
    Rs, Ss = sub(Rz), sub(Sz)

    def spread(M):
        if len(M) < 2:
            return np.nan
        d = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
        return float(d[np.triu_indices(len(M), 1)].mean())

    sr, ss = spread(Rs), spread(Ss)
    diversity = float(ss / sr) if sr and np.isfinite(sr) and sr > 0 else np.nan

    inner = NearestNeighbors(n_neighbors=2).fit(Rz).kneighbors(Rs, 2)[0][:, 1]
    radius = float(np.quantile(inner, 0.01))
    cover_r = float(np.median(inner))

    d_s2r = NearestNeighbors(n_neighbors=1).fit(Rz).kneighbors(Ss, 1)[0].ravel()
    d_r2s = NearestNeighbors(n_neighbors=1).fit(Sz).kneighbors(Rs, 1)[0].ravel()

    return Diagnostics(
        marginal_similarity=marginal,
        diversity_ratio=diversity,
        duplication_rate=float((d_s2r <= radius).mean()),
        coverage=float((d_r2s <= cover_r).mean()) if cover_r > 0 else np.nan,
    )
