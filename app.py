"""
AG-SLI Results Explorer — Streamlit demo
-----------------------------------------
Visualizes the convergence-speed comparison between baseline Stroke Location
Inversion (SLI) and Attention-Guided SLI (AG-SLI), for the SketchXAI project.

Usage:
    streamlit run app.py

By default this runs on built-in demo numbers (taken from the notebook's
confirmatory experiments) so it works out of the box with no extra files.
To show your own results, export a small JSON from the notebook like:

    import json
    json.dump({
        "baseline": {"median_iters": 15, "mean_iters": 34.12, "never_converged_pct": 21.1},
        "agsli":    {"median_iters": 10, "mean_iters": 32.04, "never_converged_pct": 22.2},
        "n_samples": 90
    }, open("results.json", "w"))

and place `results.json` next to this file — it will be picked up automatically.
"""

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AG-SLI Results Explorer", page_icon="✏️", layout="centered")

DEMO_RESULTS = {
    "baseline": {"median_iters": 15, "mean_iters": 34.12, "never_converged_pct": 21.1},
    "agsli": {"median_iters": 10, "mean_iters": 32.04, "never_converged_pct": 22.2},
    "n_samples": 90,
}

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


@st.cache_data
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f), True
    return DEMO_RESULTS, False


results, is_real = load_results()

st.title("✏️ AG-SLI: Attention-Guided Stroke Location Inversion")
st.caption("SketchXAI explainability project — Muhammad Awais, University of Debrecen")

if not is_real:
    st.info(
        "Showing **demo numbers** from the notebook's confirmatory run "
        "(90 independently-sampled 'hard' test cases). Drop a `results.json` "
        "next to this file to show your own results — see the docstring at "
        "the top of `app.py`.",
        icon="ℹ️",
    )

st.subheader("Convergence speed: hard cases only")
st.write(
    f"'Hard cases' are test sketches where baseline SLI needs several iterations "
    f"to reach 90% confidence — the cases where a speedup actually matters. "
    f"n = **{results['n_samples']}** samples."
)

df = pd.DataFrame(
    {
        "Method": ["Baseline SLI", "AG-SLI (rank-based)"],
        "Median iters to 90% conf.": [
            results["baseline"]["median_iters"],
            results["agsli"]["median_iters"],
        ],
        "Mean iters to 90% conf.": [
            results["baseline"]["mean_iters"],
            results["agsli"]["mean_iters"],
        ],
        "Never converged (%)": [
            results["baseline"]["never_converged_pct"],
            results["agsli"]["never_converged_pct"],
        ],
    }
)

st.dataframe(df, hide_index=True, use_container_width=True)

st.bar_chart(df.set_index("Method")[["Median iters to 90% conf."]])

improvement = (
    (results["baseline"]["median_iters"] - results["agsli"]["median_iters"])
    / results["baseline"]["median_iters"]
    * 100
)
st.metric(
    "Median convergence speedup on hard cases",
    f"{improvement:.1f}%",
    help="(baseline median iters − AG-SLI median iters) / baseline median iters",
)

st.divider()
st.subheader("About AG-SLI")
st.markdown(
    """
AG-SLI reconstructs plausible stroke positions for a target sketch category by optimizing
random stroke coordinates against a trained ViT-based classifier. The **rank-based** mechanism
boosts the gradient of the top 30% most-attended strokes (by the model's own final-layer
attention, averaged over the last 4 transformer layers) by 2×, linearly decaying to 1× over the
first half of optimization — consistently speeding up convergence on hard cases without hurting
final recovery confidence.

An earlier **continuous** attention-weighted scaling formulation was also tried, but honestly
did not hold up at scale — this negative result is documented in the full notebook rather than
hidden.
"""
)

st.caption("Full training + experiment notebook: `ag-sli-clean.ipynb` in this repository.")
