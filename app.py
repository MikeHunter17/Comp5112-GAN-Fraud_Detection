"""Streamlit GUI for the GAN augmentation pipeline. Works with any CSV."""

import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fraudgan import (Cleaner, Config, Dataset, TabularGAN, cap_majority,
                      diagnose, make_demo, run, select_features, split,
                      summarise, table, vs_baseline)

NAVY, RED, GREY = "#0C2340", "#C8102E", "#5A6875"

st.set_page_config(page_title="GAN Fraud Detection", layout="wide")

for k in ("ds", "prep", "gan", "synth", "results"):
    st.session_state.setdefault(k, None)


def chart_losses(history):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history.epoch, history.d_loss, label="Discriminator", lw=1.4, color=NAVY)
    ax.plot(history.epoch, history.g_loss, label="Generator", lw=1.4, color=RED)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def chart_conditions(results, metric, clf):
    d = results[results.classifier == clf]
    g = d.groupby("condition")[metric].agg(["mean", "std"]).reset_index().sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colours = [RED if "gan" in c else NAVY for c in g.condition]
    ax.barh(g.condition, g["mean"], xerr=g["std"].fillna(0),
            color=colours, height=.6, capsize=4)
    ax.set_xlabel(metric.replace("test_", "").upper())
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def chart_marginals(real, synth, cols):
    cols = cols[:4]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 2.8))
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.hist(real[c], bins=40, alpha=.6, density=True, label="real", color=NAVY)
        ax.hist(synth[c], bins=40, alpha=.6, density=True, label="synthetic", color=RED)
        ax.set_title(c, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ======================================================================
# sidebar: data
# ======================================================================

st.sidebar.title("Data")
source = st.sidebar.radio("Source", ["Upload CSV", "Built-in demo"], index=0)

if source == "Upload CSV":
    upload = st.sidebar.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        raw = pd.read_csv(upload)
        st.sidebar.success(f"{len(raw):,} rows, {raw.shape[1]} columns")

        target = st.sidebar.selectbox(
            "Label column", raw.columns,
            index=len(raw.columns) - 1,
            help="The column holding 0 for normal and 1 for the rare class.",
        )
        auto_cat = [c for c in raw.columns
                    if c != target and (raw[c].dtype == object or raw[c].nunique() <= 20)]
        cats = st.sidebar.multiselect("Categorical columns",
                                      [c for c in raw.columns if c != target],
                                      default=auto_cat)
        name = st.sidebar.text_input("Dataset name", upload.name.replace(".csv", ""))

        if st.sidebar.button("Load", type="primary", use_container_width=True):
            vals = sorted(raw[target].dropna().unique())
            if len(vals) != 2:
                st.sidebar.error(f"'{target}' has {len(vals)} distinct values. Needs exactly 2.")
            else:
                frame = raw.copy()
                frame[target] = (frame[target] == vals[-1]).astype(int)
                st.session_state.ds = Dataset(frame, target=target,
                                              categorical=cats, name=name)
                st.session_state.update(prep=None, gan=None, synth=None, results=None)
else:
    n = st.sidebar.slider("Rows", 5_000, 60_000, 20_000, step=5_000)
    rate = st.sidebar.slider("Minority rate (%)", 0.1, 5.0, 1.0, step=0.1)
    if st.sidebar.button("Load demo", type="primary", use_container_width=True):
        st.session_state.ds = make_demo(n=n, fraud_rate=rate / 100, seed=0)
        st.session_state.update(prep=None, gan=None, synth=None, results=None)

ds = st.session_state.ds

st.title("Generative Adversarial Networks for Class-Imbalanced Detection")
st.caption("Omoniyi (1340808) · Fijabi (1298724) — COMP 5112, Lakehead University")

if ds is None:
    st.info("Choose a data source in the sidebar to begin. "
            "The built-in demo needs no files and runs in under a minute.")
    st.stop()

# ======================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["1 · Data", "2 · Train the GAN", "3 · Generated data", "4 · Comparison"])

# ---------------------------------------------------------------- data
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(ds.frame):,}")
    c2.metric("Features", len(ds.features))
    c3.metric("Minority class", f"{ds.n_fraud:,}")
    c4.metric("Imbalance", f"{100 * ds.n_fraud / len(ds.frame):.3f}%")

    st.dataframe(ds.frame.head(8), use_container_width=True)

    st.subheader("Split and preprocess")
    a, b, c = st.columns(3)
    seed = a.number_input("Seed", 0, 999, 0)
    cap = b.number_input("Cap majority rows in training", 10_000, 2_000_000,
                         200_000, step=10_000,
                         help="Training split only. Validation and test are never capped.")
    kfeat = c.number_input("Max features", 5, 400, 50)

    budget = st.slider(
        "Minority rows available to the generator", 25, max(26, ds.n_fraud),
        min(ds.n_fraud, 400), step=25,
        help="Subsamples the minority class before splitting. Augmentation only "
             "matters when real examples are scarce, so lowering this is what "
             "makes the comparison informative.")

    if st.button("Run preprocessing", type="primary"):
        with st.spinner("Splitting and selecting features..."):
            frame = ds.frame
            if budget < ds.n_fraud:
                keep_min = frame[frame[ds.target] == 1].sample(budget, random_state=seed)
                frame = pd.concat([frame[frame[ds.target] == 0], keep_min])
                frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

            work = Dataset(frame, ds.target, ds.categorical, ds.name, ds.time_col)
            sp = split(work, seed=seed)
            train = cap_majority(sp.train, work.target, cap, seed)
            num, cat, imp = select_features(
                train, work.target, work.categorical, k=kfeat, seed=seed,
                exclude=[work.time_col] if work.time_col else [])
            cleaner = Cleaner(num, cat, work.target)
            tr = cleaner.fit_transform(train)
            st.session_state.prep = {
                "work": work, "splits": sp, "num": num, "cat": cat,
                "cleaner": cleaner, "tr": tr, "importance": imp, "seed": seed,
                "fraud": tr[tr[work.target] == 1].drop(columns=[work.target]),
            }
            st.session_state.update(gan=None, synth=None, results=None)

    if st.session_state.prep:
        p = st.session_state.prep
        st.success(f"{len(p['num'])} numeric + {len(p['cat'])} categorical features · "
                   f"{len(p['fraud']):,} minority rows for the generator")
        left, right = st.columns([1, 1])
        left.dataframe(p["splits"].counts(), use_container_width=True, hide_index=True)
        right.dataframe(p["importance"].head(8), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- gan
with tab2:
    if not st.session_state.prep:
        st.warning("Run preprocessing on the Data tab first.")
    else:
        p = st.session_state.prep
        st.markdown(
            "The **Generator** turns random noise into a candidate row. The "
            "**Discriminator** scores rows as real or generated. Each improves by "
            "defeating the other; only the Generator is kept."
        )
        a, b, c, d = st.columns(4)
        epochs = a.number_input("Epochs", 20, 2000, 300, step=20)
        zdim = b.number_input("Noise dimension", 16, 512, 128, step=16)
        batch = c.number_input("Batch size", 16, 1024, 256, step=16)
        ncrit = d.number_input("Critic steps", 1, 5, 2)

        if st.button("Train", type="primary"):
            bar = st.progress(0.0, "starting")
            t0 = time.time()
            gan = TabularGAN(z_dim=zdim, batch_size=batch, epochs=epochs,
                             n_critic=ncrit, seed=p["seed"], verbose=False)
            gan.fit(p["fraud"], numeric=p["num"], categorical=p["cat"])
            bar.progress(1.0, f"done in {time.time() - t0:.0f}s")
            st.session_state.gan = gan
            st.session_state.synth = None

        if st.session_state.gan:
            h = st.session_state.gan.loss_history()
            st.pyplot(chart_losses(h))
            last = h.tail(1).iloc[0]
            m1, m2 = st.columns(2)
            m1.metric("Discriminator loss", f"{last.d_loss:+.4f}",
                      help="Near zero means the critic can barely separate real "
                           "from synthetic, which is the goal.")
            m2.metric("Generator loss", f"{last.g_loss:+.4f}")

# ---------------------------------------------------------------- synth
with tab3:
    if not st.session_state.gan:
        st.warning("Train the GAN on the previous tab first.")
    else:
        p = st.session_state.prep
        n_gen = st.number_input("Rows to generate", 100, 100_000, 2_000, step=100)
        if st.button("Generate", type="primary"):
            with st.spinner("Sampling..."):
                st.session_state.synth = st.session_state.gan.sample(int(n_gen))

        if st.session_state.synth is not None:
            synth = st.session_state.synth
            st.dataframe(synth.head(8), use_container_width=True)

            with st.spinner("Running diagnostics..."):
                d = diagnose(p["fraud"], synth, p["num"], p["cat"],
                             cap=800, seed=p["seed"])
            row = d.row()

            st.subheader("Generator diagnostics")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Marginal similarity", f"{row['marginal_similarity']:.3f}",
                      help="Near 1.0 means individual columns match.")
            q2.metric("Diversity ratio", f"{row['diversity_ratio']:.3f}",
                      help="Near 1.0 is healthy. Far below means mode collapse.")
            q3.metric("Duplication rate", f"{row['duplication_rate']:.3f}",
                      help="High means the generator is memorising.")
            q4.metric("Coverage", f"{row['coverage']:.3f}",
                      help="Low means it explores the wrong region.")

            verdict = row["verdict"]
            (st.success if verdict == "no obvious failure" else st.warning)(verdict)

            st.pyplot(chart_marginals(p["fraud"], synth, p["num"]))

            st.download_button("Download synthetic rows (CSV)",
                               synth.to_csv(index=False).encode(),
                               f"synthetic_{p['work'].name}.csv", "text/csv")

# ---------------------------------------------------------------- compare
with tab4:
    if not st.session_state.prep:
        st.warning("Run preprocessing on the Data tab first.")
    else:
        p = st.session_state.prep
        st.markdown(
            "Every stage is identical across conditions **except augmentation**. "
            "Any difference in the results therefore has one possible cause."
        )
        has_cat = bool(p["cat"])
        options = ["none", "smote", "tabular_gan", "ctgan"] if has_cat else \
                  ["none", "smote", "borderline_smote", "adasyn", "tabular_gan", "ctgan"]
        if has_cat:
            st.caption("Categorical columns present, so BorderlineSMOTE and ADASYN are "
                       "unavailable: imbalanced-learn has no categorical-aware form of "
                       "either, and both fall back to SMOTENC.")

        a, b = st.columns(2)
        conds = a.multiselect("Conditions", options,
                              default=["none", "smote", "tabular_gan"])
        clfs = b.multiselect("Classifiers",
                             ["logistic_regression", "random_forest"],
                             default=["random_forest"])
        c, d, e = st.columns(3)
        n_seeds = c.number_input("Seeds", 1, 10, 1,
                                 help="Repeats the whole experiment. The spread "
                                      "across seeds is what separates a result "
                                      "from noise.")
        ratio = d.slider("Augmentation ratio", 0.01, 1.0, 0.10,
                         help="Target minority-to-majority proportion after augmentation.")
        ep = e.number_input("GAN epochs", 20, 2000, 300, step=20)

        if st.button("Run comparison", type="primary"):
            if not conds or not clfs:
                st.error("Pick at least one condition and one classifier.")
            else:
                with st.spinner("Running. GAN conditions dominate the time."):
                    t0 = time.time()
                    cfg = Config(conditions=conds, classifiers=clfs,
                                 seeds=list(range(int(n_seeds))), ratio=ratio,
                                 max_majority=10 ** 9, max_features=len(p["num"]) + len(p["cat"]),
                                 gan_epochs=int(ep), ctgan_epochs=int(ep), verbose=False)
                    st.session_state.results = run(p["work"], cfg)
                    st.caption(f"finished in {time.time() - t0:.0f}s")

        if st.session_state.results is not None:
            res = st.session_state.results
            st.subheader("Results")
            st.dataframe(table(res, "test_pr_auc"), use_container_width=True, hide_index=True)

            metric = st.selectbox("Metric", ["test_pr_auc", "test_mcc", "test_recall",
                                             "test_precision", "test_f1"])
            for clf in res.classifier.unique():
                st.caption(clf)
                st.pyplot(chart_conditions(res, metric, clf))

            if "none" in res.condition.values:
                st.subheader("Against the untreated baseline")
                st.dataframe(vs_baseline(res, metric), use_container_width=True,
                             hide_index=True)
                if res.seed.nunique() < 2:
                    st.error(
                        "Only one seed was run, so the spread is zero and "
                        "`beats_noise` is True for any difference at all. It means "
                        "nothing here. Run at least three seeds before reading that "
                        "column or quoting these numbers."
                    )
                else:
                    st.caption("`beats_noise` is True only when the change exceeds the "
                               "combined spread of the two conditions. It is a screen, "
                               "not a significance test.")

            st.download_button("Download results (CSV)",
                               res.to_csv(index=False).encode(),
                               f"results_{p['work'].name}.csv", "text/csv")
