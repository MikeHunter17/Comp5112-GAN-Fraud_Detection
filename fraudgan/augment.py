"""Augmentation conditions: the only stage that differs between experiments."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .gan import CTGANModel, TabularGAN


@dataclass
class Augmented:
    table: pd.DataFrame
    synthetic: Optional[pd.DataFrame]
    n_added: int
    seconds: float
    condition: str
    meta: Dict = field(default_factory=dict)


def _needed(df: pd.DataFrame, target: str, ratio: float) -> int:
    want = int(round((df[target] == 0).sum() * ratio))
    return max(0, want - int((df[target] == 1).sum()))


def none(table, target, **_) -> Augmented:
    return Augmented(table.copy(), None, 0, 0.0, "none")


def smote(table, target, categorical, numeric=None, ratio=1.0, seed=0,
          variant="smote", k=5, **_) -> Augmented:
    from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, SMOTENC

    t0 = time.perf_counter()
    X, y = table.drop(columns=[target]), table[target].astype(int)
    n_min = int(y.sum())
    if n_min <= k:
        return Augmented(table.copy(), None, 0, time.perf_counter() - t0, variant,
                         {"skipped": f"{n_min} minority rows"})

    want = max(int(round((y == 0).sum() * ratio)), n_min)
    strategy = {0: int((y == 0).sum()), 1: want}
    cat_idx = [X.columns.get_loc(c) for c in categorical if c in X.columns]
    kk = min(k, n_min - 1)

    # imbalanced-learn has no categorical-aware BorderlineSMOTE or ADASYN,
    # so these variants collapse to SMOTENC when categoricals are present.
    substituted = bool(cat_idx) and variant != "smote"
    if cat_idx:
        s = SMOTENC(categorical_features=cat_idx, sampling_strategy=strategy,
                    random_state=seed, k_neighbors=kk)
    elif variant == "borderline_smote":
        s = BorderlineSMOTE(sampling_strategy=strategy, random_state=seed, k_neighbors=kk)
    elif variant == "adasyn":
        s = ADASYN(sampling_strategy=strategy, random_state=seed, n_neighbors=kk)
    else:
        s = SMOTE(sampling_strategy=strategy, random_state=seed, k_neighbors=kk)

    Xr, yr = s.fit_resample(X, y)
    out = pd.DataFrame(Xr, columns=X.columns)
    out[target] = np.asarray(yr).astype(int)

    added = len(out) - len(table)
    return Augmented(out.reset_index(drop=True),
                     out.iloc[len(table):].copy() if added > 0 else None,
                     added, time.perf_counter() - t0, variant,
                     {"sampler": type(s).__name__, "substituted_by_smotenc": substituted})


def _gan_condition(model_cls, label, table, target, categorical, numeric,
                   ratio, seed, epochs, batch_size, verbose, **kw) -> Augmented:
    t0 = time.perf_counter()
    need = _needed(table, target, ratio)
    fraud = table[table[target] == 1].drop(columns=[target]).reset_index(drop=True)

    if need <= 0 or len(fraud) < 8:
        return Augmented(table.copy(), None, 0, time.perf_counter() - t0, label,
                         {"skipped": f"need={need}, fraud={len(fraud)}"})

    model = model_cls(epochs=epochs, batch_size=batch_size, seed=seed, verbose=verbose)
    model.fit(fraud, numeric=numeric, categorical=categorical)

    synth = model.sample(need)
    synth[target] = 1
    out = pd.concat([table, synth[table.columns]], ignore_index=True)

    meta = {"epochs": epochs, "trained_on": len(fraud), "ratio": ratio}
    if hasattr(model, "loss_history"):
        meta["final_losses"] = model.loss_history().tail(1).to_dict("records")

    return Augmented(out, synth.reset_index(drop=True), len(synth),
                     time.perf_counter() - t0, label, meta)


def tabular_gan(table, target, categorical, numeric, ratio=1.0, seed=0,
                epochs=300, batch_size=256, verbose=False, **kw) -> Augmented:
    return _gan_condition(TabularGAN, "tabular_gan", table, target, categorical,
                          numeric, ratio, seed, epochs, batch_size, verbose)


def ctgan(table, target, categorical, numeric, ratio=1.0, seed=0,
          epochs=300, batch_size=500, verbose=False, **kw) -> Augmented:
    return _gan_condition(CTGANModel, "ctgan", table, target, categorical,
                          numeric, ratio, seed, epochs, batch_size, verbose)


CONDITIONS = {
    "none": none,
    "smote": lambda **kw: smote(variant="smote", **kw),
    "borderline_smote": lambda **kw: smote(variant="borderline_smote", **kw),
    "adasyn": lambda **kw: smote(variant="adasyn", **kw),
    "tabular_gan": tabular_gan,
    "ctgan": ctgan,
}


def apply(name: str, **kw) -> Augmented:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}")
    return CONDITIONS[name](**kw)
