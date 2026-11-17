"""Dataset loaders."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    frame: pd.DataFrame
    target: str
    categorical: List[str] = field(default_factory=list)
    name: str = "dataset"
    time_col: Optional[str] = None

    @property
    def features(self) -> List[str]:
        return [c for c in self.frame.columns if c != self.target]

    @property
    def n_fraud(self) -> int:
        return int(self.frame[self.target].sum())

    def __repr__(self) -> str:
        rate = 100 * self.n_fraud / len(self.frame)
        return (f"<{self.name}: {len(self.frame):,} rows, {len(self.features)} features, "
                f"{self.n_fraud:,} fraud ({rate:.3f}%)>")


def load_paysim(path: str) -> Dataset:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("nameOrig", "nameDest", "isFlaggedFraud") if c in df])

    # Ledger-consistency features.
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    return Dataset(df, target="isFraud", categorical=["type"],
                   name="PaySim", time_col="step")


IEEE_CAT = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
]


def load_ieee(transaction_csv: str, identity_csv: Optional[str] = None) -> Dataset:
    df = pd.read_csv(transaction_csv)
    if identity_csv:
        df = df.merge(pd.read_csv(identity_csv), how="left", on="TransactionID")
    df = df.drop(columns=["TransactionID"], errors="ignore")

    id_cat = [c for c in df.columns
              if c.lower().replace("-", "_").startswith("id_") and df[c].dtype == object]
    cats = [c for c in IEEE_CAT if c in df.columns] + id_cat

    return Dataset(df, target="isFraud", categorical=cats,
                   name="IEEE-CIS", time_col="TransactionDT")


def load_ulb(path: str) -> Dataset:
    """Optional small benchmark, useful for fast development runs."""
    df = pd.read_csv(path).rename(columns={"Class": "isFraud"})
    return Dataset(df, target="isFraud", categorical=[],
                   name="ULB", time_col="Time")


def make_demo(n: int = 20_000, fraud_rate: float = 0.01, seed: int = 0) -> Dataset:
    """Synthetic stand-in with the same shape as the real data. Testing only."""
    rng = np.random.default_rng(seed)
    n_fraud = max(10, int(n * fraud_rate))
    n_legit = n - n_fraud

    X = np.vstack([rng.normal(0, 1, (n_legit, 8)),
                   rng.normal(1.8, 1.3, (n_fraud, 8))])
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(8)])

    levels = np.array(["a", "b", "c", "d"])
    df["type"] = np.concatenate([
        rng.choice(levels, n_legit, p=[.55, .25, .15, .05]),
        rng.choice(levels, n_fraud, p=[.10, .20, .30, .40]),
    ])
    df["isFraud"] = np.r_[np.zeros(n_legit, int), np.ones(n_fraud, int)]

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return Dataset(df, target="isFraud", categorical=["type"], name="Demo")
