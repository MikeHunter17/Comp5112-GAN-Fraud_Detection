"""Tabular GAN: Generator and Discriminator (WGAN-GP), plus a CTGAN wrapper."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# column transformer
# --------------------------------------------------------------------------

@dataclass
class Transformer:
    """Table <-> matrix. Numerics min-max to [-1,1]; categoricals one-hot."""
    numeric: List[str]
    categorical: List[str]
    lo: Dict[str, float] = field(default_factory=dict)
    hi: Dict[str, float] = field(default_factory=dict)
    levels: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def cat_spans(self) -> List[Tuple[int, int]]:
        """(start, end) index of each categorical block in the matrix."""
        spans, i = [], len(self.numeric)
        for c in self.categorical:
            w = len(self.levels[c])
            spans.append((i, i + w))
            i += w
        return spans

    @property
    def width(self) -> int:
        return len(self.numeric) + sum(len(self.levels[c]) for c in self.categorical)

    def fit(self, df: pd.DataFrame) -> "Transformer":
        for c in self.numeric:
            s = pd.to_numeric(df[c], errors="coerce")
            self.lo[c], self.hi[c] = float(s.min()), float(s.max())
            if self.hi[c] <= self.lo[c]:
                self.hi[c] = self.lo[c] + 1.0
        for c in self.categorical:
            self.levels[c] = sorted(df[c].astype(str).unique())
        return self

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.numeric:
            s = pd.to_numeric(df[c], errors="coerce").fillna(self.lo[c]).to_numpy(float)
            cols.append(2 * (s - self.lo[c]) / (self.hi[c] - self.lo[c]) - 1)
        out = [np.stack(cols, 1)] if cols else []
        for c in self.categorical:
            codes = df[c].astype(str).to_numpy()
            block = np.zeros((len(df), len(self.levels[c])), np.float32)
            for j, lv in enumerate(self.levels[c]):
                block[:, j] = (codes == lv)
            out.append(block)
        return np.hstack(out).astype(np.float32)

    def decode(self, M: np.ndarray) -> pd.DataFrame:
        out = {}
        for i, c in enumerate(self.numeric):
            v = np.clip(M[:, i], -1, 1)
            out[c] = (v + 1) / 2 * (self.hi[c] - self.lo[c]) + self.lo[c]
        for c, (a, b) in zip(self.categorical, self.cat_spans):
            idx = M[:, a:b].argmax(1)
            out[c] = np.array(self.levels[c])[idx]
        return pd.DataFrame(out)


# --------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------

class Generator(nn.Module):
    """noise -> synthetic row. tanh for numerics, softmax per categorical block."""

    def __init__(self, z_dim: int, out_dim: int, n_numeric: int,
                 cat_spans: List[Tuple[int, int]], hidden=(256, 256)):
        super().__init__()
        self.n_numeric = n_numeric
        self.cat_spans = cat_spans

        layers, d = [], z_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, tau: float = 0.2, hard: bool = False) -> torch.Tensor:
        raw = self.net(z)
        parts = [torch.tanh(raw[:, :self.n_numeric])] if self.n_numeric else []
        for a, b in self.cat_spans:
            parts.append(F.gumbel_softmax(raw[:, a:b], tau=tau, hard=hard))
        return torch.cat(parts, 1) if parts else raw


class Discriminator(nn.Module):
    """row -> realness score. No sigmoid: this is a WGAN critic."""

    def __init__(self, in_dim: int, hidden=(256, 256)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gradient_penalty(critic: Discriminator, real: torch.Tensor,
                     fake: torch.Tensor, dev: torch.device) -> torch.Tensor:
    eps = torch.rand(real.size(0), 1, device=dev)
    mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(mixed)
    grad = torch.autograd.grad(score, mixed, torch.ones_like(score),
                               create_graph=True, retain_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class TabularGAN:
    """WGAN-GP for mixed-type tabular data."""

    def __init__(self, z_dim=128, hidden=(256, 256), batch_size=256,
                 epochs=300, lr=2e-4, n_critic=2, gp_weight=10.0,
                 tau=0.2, seed=0, verbose=False):
        self.z_dim, self.hidden = z_dim, hidden
        self.batch_size, self.epochs = batch_size, epochs
        self.lr, self.n_critic, self.gp_weight = lr, n_critic, gp_weight
        self.tau, self.seed, self.verbose = tau, seed, verbose
        self.tf: Optional[Transformer] = None
        self.G: Optional[Generator] = None
        self.D: Optional[Discriminator] = None
        self.history: List[Dict] = []

    def fit(self, df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> "TabularGAN":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = device()

        self.tf = Transformer(numeric, categorical).fit(df)
        X = torch.tensor(self.tf.encode(df), dtype=torch.float32, device=dev)

        bs = min(self.batch_size, max(8, len(X) // 2))
        self.G = Generator(self.z_dim, self.tf.width, len(numeric),
                           self.tf.cat_spans, self.hidden).to(dev)
        self.D = Discriminator(self.tf.width, self.hidden).to(dev)

        optG = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        optD = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

        for epoch in range(self.epochs):
            perm = torch.randperm(len(X), device=dev)
            d_running, g_running, n_batches = 0.0, 0.0, 0

            for start in range(0, len(X) - bs + 1, bs):
                real = X[perm[start:start + bs]]

                # --- critic ---
                for _ in range(self.n_critic):
                    z = torch.randn(bs, self.z_dim, device=dev)
                    with torch.no_grad():
                        fake = self.G(z, self.tau)
                    d_loss = (self.D(fake).mean() - self.D(real).mean()
                              + self.gp_weight * gradient_penalty(self.D, real, fake, dev))
                    optD.zero_grad()
                    d_loss.backward()
                    optD.step()

                # --- generator ---
                z = torch.randn(bs, self.z_dim, device=dev)
                g_loss = -self.D(self.G(z, self.tau)).mean()
                optG.zero_grad()
                g_loss.backward()
                optG.step()

                d_running += float(d_loss)
                g_running += float(g_loss)
                n_batches += 1

            if n_batches:
                self.history.append({"epoch": epoch,
                                     "d_loss": d_running / n_batches,
                                     "g_loss": g_running / n_batches})
                if self.verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                    h = self.history[-1]
                    print(f"  epoch {epoch:4d}  D {h['d_loss']:+.4f}  G {h['g_loss']:+.4f}")

        return self

    @torch.no_grad()
    def sample(self, n: int) -> pd.DataFrame:
        self.G.eval()
        dev = device()
        out = []
        for start in range(0, n, 2048):
            k = min(2048, n - start)
            z = torch.randn(k, self.z_dim, device=dev)
            out.append(self.G(z, self.tau, hard=True).cpu().numpy())
        self.G.train()
        return self.tf.decode(np.vstack(out))

    def loss_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# --------------------------------------------------------------------------
# CTGAN wrapper (library baseline)
# --------------------------------------------------------------------------

class CTGANModel:
    """Thin wrapper so CTGAN has the same fit/sample interface as TabularGAN."""

    def __init__(self, epochs=300, batch_size=500, seed=0, verbose=False):
        self.epochs, self.batch_size = epochs, batch_size
        self.seed, self.verbose = seed, verbose
        self.model = None

    def fit(self, df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> "CTGANModel":
        from ctgan import CTGAN
        bs = min(self.batch_size, max(10, (len(df) // 10) * 10))
        bs = max(10, bs - bs % 10)
        self.model = CTGAN(epochs=self.epochs, batch_size=bs, verbose=self.verbose)
        self.model.set_random_state(self.seed)
        self.model.fit(df, discrete_columns=[c for c in categorical if c in df])
        return self

    def sample(self, n: int) -> pd.DataFrame:
        return self.model.sample(n)
