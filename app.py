import numpy as np
import matplotlib.pyplot as plt

from shiny import App, ui, reactive, render, run_app


def normal_pdf(x, mu, sd):
    return 1.0 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sd) ** 2)


app_ui = ui.page_fluid(
    ui.h2("Precision–Recall Tradeoff — Two Gaussian Classes"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Controls"),
            ui.input_slider("mean1", "Mean (Class 1)", -1.0, 4.0, value=1.0, step=0.1),
            ui.input_slider("sd", "Std dev (both)", 0.1, 3.0, value=1.0, step=0.05),
            ui.input_slider("threshold", "Decision threshold", -6.0, 6.0, value=0.0, step=0.01),
            ui.input_numeric("n", "Samples per class", value=1000, min=100, max=10000, step=100),
            ui.markdown("\nChange the threshold to see how precision and recall move.\n"),
        ),
        ui.output_plot("gaussians", height="300px"),
        ui.div(
            ui.h4("Metrics"),
            ui.row(
                ui.column(4, ui.output_plot("pr_plot", height="180px")),
                ui.column(4, ui.output_plot("conf_matrix", height="250px")),
                ui.column(4, ui.output_plot("roc_curve", height="300px")),
            ),
        )
    ),
)


def make_dataset(mean0, mean1, sd, n_per_class, seed=0):
    # deterministic-ish for a given seed
    rng = np.random.default_rng(seed)
    x0 = rng.normal(mean0, sd, size=n_per_class)
    x1 = rng.normal(mean1, sd, size=n_per_class)
    X = np.concatenate([x0, x1])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y


def compute_confusion(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return np.array([[tn, fp], [fn, tp]]), tp, fp, tn, fn


def compute_roc_points(X, y, num=200):
    # thresholds spanning the data range
    thr_min, thr_max = X.min() - 1.0, X.max() + 1.0
    thresholds = np.linspace(thr_min, thr_max, num=num)
    tprs = []
    fprs = []
    for thr in thresholds:
        y_pred = (X >= thr).astype(int)
        cm, tp, fp, tn, fn = compute_confusion(y, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array(fprs), np.array(tprs), thresholds


def format_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def server(input, output, session):
    @reactive.Calc
    def dataset():
        mean0 = -1.0
        mean1 = float(input.mean1())
        sd = float(input.sd())
        n = int(input.n())
        # change seed when parameters change so visuals update deterministically per-param
        raw_seed = int((mean0 + mean1) * 100 + sd * 10)
        # ensure seed is a non-negative integer
        seed = raw_seed if raw_seed >= 0 else -raw_seed
        X, y = make_dataset(mean0, mean1, sd, n_per_class=n, seed=seed)
        return X, y, mean0, mean1, sd

    @output
    @render.plot
    def gaussians():
        X, y, mean0, mean1, sd = dataset()
        thr = float(input.threshold())

        fig, ax = plt.subplots(figsize=(7, 2.7))
        xgrid = np.linspace(X.min() - 1.0, X.max() + 1.0, 400)
        pdf0 = normal_pdf(xgrid, mean0, sd)
        pdf1 = normal_pdf(xgrid, mean1, sd)

        ax.plot(xgrid, pdf0, label="Class 0 PDF", color="#1f77b4")
        ax.plot(xgrid, pdf1, label="Class 1 PDF", color="#ff7f0e")

        # show histogram (density) of samples to illustrate overlap
        ax.hist(X[y == 0], bins=60, density=True, alpha=0.15, color="#1f77b4")
        ax.hist(X[y == 1], bins=60, density=True, alpha=0.15, color="#ff7f0e")

        ax.axvline(thr, color="k", linestyle="--", label=f"threshold = {thr:.2f}")
        ax.legend(loc="upper right")
        ax.set_title("Class PDFs and decision threshold")
        ax.set_xlabel("x")
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.18)
        return fig

    @output
    @render.plot
    def pr_plot():
        X, y, mean0, mean1, sd = dataset()
        thr = float(input.threshold())
        y_pred = (X >= thr).astype(int)
        cm, tp, fp, tn, fn = compute_confusion(y, y_pred)
        precision, recall = format_precision_recall(tp, fp, fn)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter([recall], [precision], c=["red"], s=80)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Recall")
        # annotate counts in axes fraction so it doesn't overlap the point
        ax.annotate(f"TP={tp} FP={fp}\nFN={fn} TN={tn}", xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.2))
        ax.grid(alpha=0.3)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12)
        return fig

    @output
    @render.plot
    def conf_matrix():
        X, y, mean0, mean1, sd = dataset()
        thr = float(input.threshold())
        y_pred = (X >= thr).astype(int)
        cm, tp, fp, tn, fn = compute_confusion(y, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        mat = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        vmax = mat.max() if mat.size > 0 else 1
        for i in range(2):
            for j in range(2):
                # choose text color for readability depending on background intensity
                text_color = "white" if mat[i, j] > vmax / 2 else "black"
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", color=text_color, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True 0", "True 1"]) 
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12)
        return fig

    @output
    @render.plot
    def roc_curve():
        X, y, mean0, mean1, sd = dataset()
        fprs, tprs, thresholds = compute_roc_points(X, y, num=300)
        thr = float(input.threshold())
        # compute current point
        y_pred = (X >= thr).astype(int)
        cm, tp, fp, tn, fn = compute_confusion(y, y_pred)
        cur_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cur_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fprs, tprs, label="ROC curve")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.scatter([cur_fpr], [cur_tpr], color="red", zorder=5, label=f"threshold: {thr:.2f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve")
        ax.legend()
        fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12)
        return fig


app = App(app_ui, server)


if __name__ == "__main__":
    # run with: python week8/app.py
    run_app(app, port=8000, host="127.0.0.1")
