"""Experiment runner and results aggregation."""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from . import augment as aug
from .data import Dataset
from .evaluate import diagnose, fit_and_score
from .prep import Cleaner, Encoder, cap_majority, select_features, split, split_by_time


@dataclass
class Config:
    conditions: List[str] = field(default_factory=lambda: ["none", "smote", "tabular_gan", "ctgan"])
    classifiers: List[str] = field(default_factory=lambda: ["logistic_regression", "random_forest"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    ratio: float = 0.10
    max_majority: int = 200_000
    max_features: int = 50
    gan_epochs: int = 300
    gan_batch: int = 256
    ctgan_epochs: int = 300
    ctgan_batch: int = 500
    temporal: bool = False
    diagnostics: bool = True
    verbose: bool = True


def run(ds: Dataset, cfg: Optional[Config] = None) -> pd.DataFrame:
    cfg = cfg or Config()
    rows = []

    if cfg.verbose:
        print(ds)

    for seed in cfg.seeds:
        t0 = time.perf_counter()
        sp = split_by_time(ds) if cfg.temporal else split(ds, seed=seed)
        train = cap_majority(sp.train, ds.target, cfg.max_majority, seed)

        num, cat, _ = select_features(train, ds.target, ds.categorical,
                                      k=cfg.max_features, seed=seed,
                                      exclude=[ds.time_col] if ds.time_col else [])

        cleaner = Cleaner(num, cat, ds.target)
        tr_tbl = cleaner.fit_transform(train)
        va_tbl = cleaner.transform(sp.val)
        te_tbl = cleaner.transform(sp.test)
        real_fraud = tr_tbl[tr_tbl[ds.target] == 1].drop(columns=[ds.target])

        if cfg.verbose:
            print(f"\nseed {seed}: train {len(tr_tbl):,} "
                  f"({int(tr_tbl[ds.target].sum()):,} fraud), "
                  f"{len(num)}+{len(cat)} features")

        for cond in cfg.conditions:
            is_ctgan = cond == "ctgan"
            a = aug.apply(cond, table=tr_tbl, target=ds.target, categorical=cat,
                          numeric=num, ratio=cfg.ratio, seed=seed,
                          epochs=cfg.ctgan_epochs if is_ctgan else cfg.gan_epochs,
                          batch_size=cfg.ctgan_batch if is_ctgan else cfg.gan_batch,
                          verbose=False)

            diag = {}
            if cfg.diagnostics and a.synthetic is not None and len(a.synthetic):
                d = diagnose(real_fraud,
                             a.synthetic.drop(columns=[ds.target], errors="ignore"),
                             num, cat, seed=seed)
                diag = d.row()

            enc = Encoder(num, cat, cleaner.levels)
            Xtr = enc.fit_transform(a.table)
            ytr = a.table[ds.target].to_numpy()
            Xva, yva = enc.transform(va_tbl), va_tbl[ds.target].to_numpy()
            Xte, yte = enc.transform(te_tbl), te_tbl[ds.target].to_numpy()

            for clf in cfg.classifiers:
                va_s, te_s, secs = fit_and_score(clf, Xtr, ytr, Xva, yva, Xte, yte, seed)
                row = {"dataset": ds.name, "seed": seed, "condition": cond,
                       "classifier": clf, "n_added": a.n_added,
                       "augment_s": round(a.seconds, 2), "fit_s": round(secs, 2),
                       "total_s": round(a.seconds + secs, 2)}
                row.update({f"test_{k}": v for k, v in te_s.row().items()})
                row.update(diag)
                rows.append(row)

                if cfg.verbose:
                    print(f"   {cond:<18}{clf:<22}"
                          f"PR-AUC {te_s.pr_auc:.4f}  MCC {te_s.mcc:.4f}  "
                          f"rec {te_s.recall:.3f}  prec {te_s.precision:.3f}  "
                          f"{a.seconds + secs:6.1f}s")

        if cfg.verbose:
            print(f"   seed done in {time.perf_counter() - t0:.0f}s")

    return pd.DataFrame(rows)


METRICS = ["test_pr_auc", "test_mcc", "test_recall", "test_precision", "test_f1"]


def summarise(results: pd.DataFrame, metrics=None) -> pd.DataFrame:
    metrics = metrics or METRICS
    keys = ["dataset", "condition", "classifier"]
    out = results.groupby(keys)[metrics + ["total_s"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    out["n_seeds"] = results.groupby(keys).size().values
    return out


def table(results: pd.DataFrame, metric="test_pr_auc") -> pd.DataFrame:
    """Paper table: mean +/- sd per condition, datasets across the top."""
    s = summarise(results, [metric])
    m, sd = f"{metric}_mean", f"{metric}_std"
    s["cell"] = s.apply(lambda r: f"{r[m]:.4f} ± {0 if pd.isna(r[sd]) else r[sd]:.4f}", axis=1)
    return s.pivot_table(index=["condition", "classifier"], columns="dataset",
                         values="cell", aggfunc="first").reset_index()


def vs_baseline(results: pd.DataFrame, metric="test_pr_auc", base="none") -> pd.DataFrame:
    """Change against the untreated baseline, screened against seed noise."""
    s = summarise(results, [metric])
    m, sd = f"{metric}_mean", f"{metric}_std"
    out = []
    for (d, c), blk in s.groupby(["dataset", "classifier"]):
        b = blk[blk.condition == base]
        if b.empty:
            continue
        bm, bs = float(b[m].iloc[0]), float(b[sd].fillna(0).iloc[0])
        for _, r in blk[blk.condition != base].iterrows():
            delta = float(r[m]) - bm
            noise = bs + float(0 if pd.isna(r[sd]) else r[sd])
            out.append({"dataset": d, "classifier": c, "condition": r.condition,
                        "baseline": round(bm, 4), "mean": round(float(r[m]), 4),
                        "delta": round(delta, 4), "noise": round(noise, 4),
                        "beats_noise": abs(delta) > noise})
    return pd.DataFrame(out)
