import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. Config
# =========================================================
BASE_ROOT = Path("/Users/xluan3/ASU Dropbox/Xinmiao Luan/Semiconductor_Project/result")

METHOD_DIRS = {
    "baseline1":  BASE_ROOT / "baseline1",
    "baseline2":  BASE_ROOT / "baseline2",
    "baseline3":  BASE_ROOT / "baseline3",
    # "our method": BASE_ROOT / "router_global_patch_cls_result",
    "our method":  BASE_ROOT / "ensemble",
    # ── ResNet-50 baselines ──────────────────────────────────
    "B4 ResNet+LR":      BASE_ROOT / "baseline4",
    "B5 ResNet+kNN":     BASE_ROOT / "baseline5",
    "B6 ResNet+routing": BASE_ROOT / "baseline6",
}

OUT_DIR = BASE_ROOT / "roc_pr_plots_from_csv_updated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_CLASSES = [
    "defect1", "defect2", "defect3",
    "defect4", "defect5", "defect8",
    "defect9", "defect10"
]

# Color per method — DINOv2 methods in cool blues/greens/reds,
# ResNet methods in warm oranges/browns (same hue family as their DINOv2 counterpart)
METHOD_COLORS = {
    "our method":        "#E53935",   # red solid — best
    "baseline1":         "#1E88E5",   # blue dashed
    "baseline2":         "#43A047",   # green dashed
    "baseline3":         "#FB8C00",   # orange dashed
    "B6 ResNet+routing": "#E53935",   # red dashed — same family as ours
    "B4 ResNet+LR":      "#1E88E5",   # blue dotted — same family as B1
    "B5 ResNet+kNN":     "#43A047",   # green dotted — same family as B2/B3
}


# =========================================================
# 2. Helper functions
# =========================================================
def find_csv_files(method_dir: Path):
    roc_files     = list(method_dir.rglob("roc_curve_points.csv"))
    pr_files      = list(method_dir.rglob("pr_curve_points.csv"))
    summary_files = list(method_dir.rglob("roc_pr_summary.csv"))

    if not roc_files:
        raise FileNotFoundError(f"roc_curve_points.csv not found: {method_dir}")

    roc_path     = max(roc_files,     key=lambda p: p.stat().st_mtime)
    pr_path      = max(pr_files,      key=lambda p: p.stat().st_mtime)
    summary_path = max(summary_files, key=lambda p: p.stat().st_mtime)

    return roc_path, pr_path, summary_path


def load_method_csvs(method_dir: Path):
    roc_path, pr_path, summary_path = find_csv_files(method_dir)
    return {
        "roc":     pd.read_csv(roc_path),
        "pr":      pd.read_csv(pr_path),
        "summary": pd.read_csv(summary_path),
    }


def get_metric(df_summary, class_name, metric):
    row = df_summary[df_summary["class"] == class_name]
    if len(row) == 0:
        return None
    val = row.iloc[0][metric]
    return None if pd.isna(val) else float(val)


def get_linestyle(method_name):
    """
    our method        → solid
    B6 ResNet+routing → dashed  (same routing strategy, different backbone)
    all other DINOv2  → dashed
    all other ResNet  → dotted
    """
    if method_name == "our method":
        return "-"
    if method_name.startswith("B") and "ResNet" in method_name:
        return ":"
    return "--"


def get_color(method_name):
    return METHOD_COLORS.get(method_name, "#888888")


def get_linewidth(method_name):
    return 2.2 if method_name in ("our method", "B6 ResNet+routing") else 1.4


# =========================================================
# 3. Micro-average plots
# =========================================================
def plot_micro_roc(method_results):
    plt.figure(figsize=(7, 6))

    for method_name, res in method_results.items():
        sub = res["roc"][res["roc"]["class"] == "micro"]
        if sub.empty:
            continue
        auc = get_metric(res["summary"], "micro", "roc_auc")
        label = f"{method_name} (AUC={auc:.3f})" if auc is not None else method_name
        plt.plot(
            sub["fpr"], sub["tpr"],
            linestyle=get_linestyle(method_name),
            linewidth=get_linewidth(method_name),
            color=get_color(method_name),
            label=label,
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC Comparison")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "micro_roc_comparison.png", dpi=300)
    plt.close()
    print("Saved: micro_roc_comparison.png")


def plot_micro_pr(method_results):
    plt.figure(figsize=(7, 6))

    for method_name, res in method_results.items():
        sub = res["pr"][res["pr"]["class"] == "micro"]
        if sub.empty:
            continue
        ap = get_metric(res["summary"], "micro", "average_precision")
        label = f"{method_name} (AP={ap:.3f})" if ap is not None else method_name
        plt.plot(
            sub["recall"], sub["precision"],
            linestyle=get_linestyle(method_name),
            linewidth=get_linewidth(method_name),
            color=get_color(method_name),
            label=label,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-average PR Comparison")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "micro_pr_comparison.png", dpi=300)
    plt.close()
    print("Saved: micro_pr_comparison.png")


# =========================================================
# 4. Per-defect plots
# =========================================================
def plot_defect_roc(defect, method_results):
    plt.figure(figsize=(7, 6))
    plotted = False

    for method_name, res in method_results.items():
        sub = res["roc"][res["roc"]["class"] == defect]
        if sub.empty:
            continue
        auc = get_metric(res["summary"], defect, "roc_auc")
        label = f"{method_name} (AUC={auc:.3f})" if auc is not None else method_name
        plt.plot(
            sub["fpr"], sub["tpr"],
            linestyle=get_linestyle(method_name),
            linewidth=get_linewidth(method_name),
            color=get_color(method_name),
            label=label,
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Comparison - {defect}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{defect}_roc_comparison.png", dpi=300)
    plt.close()


def plot_defect_pr(defect, method_results):
    plt.figure(figsize=(7, 6))
    plotted = False

    for method_name, res in method_results.items():
        sub = res["pr"][res["pr"]["class"] == defect]
        if sub.empty:
            continue
        ap = get_metric(res["summary"], defect, "average_precision")
        label = f"{method_name} (AP={ap:.3f})" if ap is not None else method_name
        plt.plot(
            sub["recall"], sub["precision"],
            linestyle=get_linestyle(method_name),
            linewidth=get_linewidth(method_name),
            color=get_color(method_name),
            label=label,
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Comparison - {defect}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{defect}_pr_comparison.png", dpi=300)
    plt.close()


# =========================================================
# 5. Main
# =========================================================
def main():
    method_results = {}

    for method_name, method_dir in METHOD_DIRS.items():
        if not method_dir.exists():
            print(f"skip (not found): {method_name}  →  {method_dir}")
            continue
        try:
            method_results[method_name] = load_method_csvs(method_dir)
            print(f"Loaded: {method_name}")
        except Exception as e:
            print(f"Fail:   {method_name}  —  {e}")

    if not method_results:
        print("No data found.")
        return

    plot_micro_roc(method_results)
    plot_micro_pr(method_results)

    for defect in DEFECT_CLASSES:
        plot_defect_roc(defect, method_results)
        plot_defect_pr(defect, method_results)

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()