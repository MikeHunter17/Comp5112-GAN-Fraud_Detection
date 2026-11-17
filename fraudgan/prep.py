"""Splitting, feature selection and encoding."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import Dataset

MISSING = "__missing__"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target: str

    def counts(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"split": k, "rows": len(d), "fraud": int(d[self.target].sum()),
             "rate_%": round(100 * d[self.target].mean(), 4)}
            for k, d in (("train", self.train), ("val", self.val), ("test", self.test))
        ])

    @property
    def train_fraud(self) -> pd.DataFrame:
        return self.train[self.train[self.target] == 1]


def split(ds: Dataset, train_frac=0.70, val_frac=0.15, seed=0) -> Splits:
    """Stratified 70/15/15. Runs before anything is fitted."""
    y = ds.frame[ds.target]
    tr, rest = train_test_split(ds.frame, train_size=train_frac,
                                stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=val_frac / (1 - train_frac),
                              stratify=rest[ds.target], random_state=seed)
    return Splits(tr.reset_index(drop=True), va.reset_index(drop=True),
                  te.reset_index(drop=True), ds.target)


def split_by_time(ds: Dataset, train_frac=0.70, val_frac=0.15) -> Splits:
    """Chronological split. Robustness check only."""
    df = ds.frame.sort_values(ds.time_col).reset_index(drop=True)
    i, j = int(len(df) * train_frac), int(len(df) * (train_frac + val_frac))
    return Splits(df.iloc[:i].reset_index(drop=True),
                  df.iloc[i:j].reset_index(drop=True),
                  df.iloc[j:].reset_index(drop=True), ds.target)


def cap_majority(df: pd.DataFrame, target: str, limit: int, seed=0) -> pd.DataFrame:
    """Cap legitimate rows in the training split. Train only, never test."""
    maj = df[df[target] == 0]
    if len(maj) <= limit:
        return df
    keep = pd.concat([maj.sample(limit, random_state=seed), df[df[target] == 1]])
    return keep.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def select_features(train: pd.DataFrame, target: str, categorical: List[str],
                    k: int = 50, max_missing=0.6, max_levels=50,
                    seed=0, exclude: Optional[List[str]] = None
                    ) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Drop sparse and high-cardinality columns, then keep the top k by RF importance."""
    skip = set(exclude or []) | {target}
    cols = [c for c in train.columns if c not in skip]

    missing = train[cols].isna().mean()
    cols = [c for c in cols if missing[c] <= max_missing]

    cat = [c for c in cols if c in categorical and train[c].nunique() <= max_levels]
    num = [c for c in cols if c not in categorical]
    if not (num or cat):
        raise ValueError("feature selection removed everything")

    rank = pd.DataFrame(index=train.index)
    for c in num:
        s = pd.to_numeric(train[c], errors="coerce")
        rank[c] = s.fillna(s.median())
    for c in cat:
        rank[c] = train[c].astype("category").cat.codes

    rf = RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=-1,
                                random_state=seed, class_weight="balanced_subsample")
    rf.fit(rank, train[target])

    imp = (pd.DataFrame({"feature": rank.columns, "importance": rf.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    keep = set(imp.head(k).feature)

    return [c for c in num if c in keep], [c for c in cat if c in keep], imp


@dataclass
class Cleaner:
    """Fills gaps and fixes category vocabularies. Keeps table shape for the GAN."""
    numeric: List[str]
    categorical: List[str]
    target: str
    medians: Dict[str, float] = field(default_factory=dict)
    levels: Dict[str, List[str]] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "Cleaner":
        for c in self.numeric:
            s = pd.to_numeric(train[c], errors="coerce").median()
            self.medians[c] = 0.0 if pd.isna(s) else float(s)
        for c in self.categorical:
            s = train[c].astype(object).where(train[c].notna(), MISSING).astype(str)
            self.levels[c] = sorted(s.unique())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.numeric:
            s = pd.to_numeric(df[c], errors="coerce")
            out[c] = s.fillna(self.medians[c]).astype(float)
        for c in self.categorical:
            s = df[c].astype(object).where(df[c].notna(), MISSING).astype(str)
            out[c] = s.where(s.isin(set(self.levels[c])), MISSING)
        if self.target in df.columns:
            out[self.target] = df[self.target].astype(int).values
        return out.reset_index(drop=True)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)


@dataclass
class Encoder:
    """One-hot categoricals, standardise numerics. Produces the classifier matrix."""
    numeric: List[str]
    categorical: List[str]
    levels: Dict[str, List[str]]
    scaler: Optional[StandardScaler] = None
    columns: List[str] = field(default_factory=list)

    def _wide(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [df[self.numeric].astype(float).reset_index(drop=True)] if self.numeric else []
        for c in self.categorical:
            codes = df[c].astype(str).reset_index(drop=True)
            parts.append(pd.DataFrame({f"{c}={v}": (codes == v).astype(np.int8)
                                       for v in self.levels[c]}))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    def fit(self, df: pd.DataFrame) -> "Encoder":
        wide = self._wide(df)
        self.columns = list(wide.columns)
        if self.numeric:
            self.scaler = StandardScaler().fit(wide[self.numeric].values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        wide = self._wide(df)
        for c in self.columns:
            if c not in wide:
                wide[c] = 0
        wide = wide[self.columns]
        if self.scaler is not None:
            wide[self.numeric] = self.scaler.transform(wide[self.numeric].values)
        return wide.to_numpy(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
