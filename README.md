# SketchXAI — Attention-Guided Stroke Location Inversion (AG-SLI)

Explainable AI for human sketch recognition. AG-SLI speeds up **Stroke Location Inversion (SLI)** —
a technique that reconstructs plausible stroke positions from a trained sketch classifier — by
guiding the optimization with the model's own transformer attention over strokes.

Built on top of the ViT-based [SketchXAI](https://github.com/WinKawaks/SketchXAI) architecture.

## Highlights

- **86.13% validation accuracy** training a ViT-based sketch classifier from scratch on a
  90,000-sketch QuickDraw subset (30 categories), in 10 epochs.
- Two AG-SLI formulations, evaluated honestly:
  - **Continuous attention-weighted gradient scaling** (exploratory) — tested at scale and found to
    give **no consistent** convergence-speed benefit.
  - **Rank-based top-k attention boost** (refined) — boosts the gradient of the top 30% most-attended
    strokes (averaged over the last 4 transformer layers) by a fixed 2× factor, linearly decaying to
    1× over the first half of optimization. This mechanism **consistently and reproducibly** improves
    convergence speed on independently-sampled hard cases, without hurting final recovery accuracy.
- Qualitative recovery panels comparing baseline vs. AG-SLI stroke reconstruction over iterations.

## Repository structure

```
├── ag-sli-clean.ipynb     # Main notebook: training + AG-SLI experiments, cleaned & documented
├── app.py                 # Streamlit demo (see below)
├── requirements.txt
└── README.md
```

## Method summary

Given a trained sketch classifier and a target category, SLI starts from randomized stroke
positions and optimizes them (via gradient descent) to maximize the classifier's confidence in the
target category — effectively "inverting" the model to reveal what stroke arrangement it associates
with a class. AG-SLI accelerates this by scaling gradients according to which strokes the model's
own attention says matter most for the current prediction.

## Running the notebook

The notebook was developed on Kaggle/Colab with a Tesla T4 GPU. It expects:
- A QuickDraw-derived HDF5 dataset with a `categories.pkl` index file (30 categories)
- The [SketchXAI](https://github.com/WinKawaks/SketchXAI) repo cloned alongside (done automatically
  in the first cell)

Open `ag-sli-clean.ipynb` and run top to bottom. Training takes several hours on a single T4; the
inversion/AG-SLI experiments run in minutes.

## Streamlit demo

`app.py` is a lightweight interactive demo that visualizes the convergence-speed comparison between
baseline SLI and AG-SLI (using the results produced by the notebook), so recruiters/reviewers can
explore the results without running the notebook. See inline comments in `app.py` for how to point
it at your own results file.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Citation

If the original SketchXAI architecture or SLI mechanism is referenced, please cite the original paper:

```
@article{qu2023sketchxai,
  title={SketchXAI: A First Look at Explainability for Human Sketches},
  author={Qu, Zhiyu and others},
  year={2023}
}
```

## Author

Muhammad Awais — PhD Candidate, University of Debrecen, Hungary.
Work submitted to *Scientific Reports* (Springer Nature), currently under peer review.
