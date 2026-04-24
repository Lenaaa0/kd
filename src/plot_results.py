"""
可视化脚本：基于 runs/experiment_*/summary.json 生成论文图表
用法: python src/plot_results.py --run_dir runs/experiment_20260423_233453

所有图均基于真实实验数据：
  - fig1: 模型准确率 / F1 / 延迟对比柱状图
  - fig2: 蒸馏 CNN 混淆矩阵（从模型权重真实推理）
  - fig3: LR 系数热力图（可解释性）
  - fig4: t-SNE 特征空间可视化
  - fig5: 训练收敛曲线（验证集，不是测试集）
  - fig6: 消融实验（T 和 α 敏感性，完整 150 epochs）
"""

import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── 配色方案 ──────────────────────────────────────────────────────────
COLORS = {
    "LR":              "#94a3b8",
    "Pure CNN":        "#f97316",
    "Teacher":         "#2563eb",
    "Distill CNN":     "#16a34a",
    "Features+LR":     "#7c3aed",
}
KEY_MAP = {
    "1-LR":           ("LR",             COLORS["LR"]),
    "2-CNN":          ("Pure CNN",        COLORS["Pure CNN"]),
    "3-Teacher":       ("Teacher",         COLORS["Teacher"]),
    "4-DistillCNN":   ("Distill CNN",    COLORS["Distill CNN"]),
    "5-Features+LR":  ("CNN Features+LR",COLORS["Features+LR"]),
}

# ── 字体配置 ──────────────────────────────────────────────────────────
def setup_fonts():
    font_list = matplotlib.font_manager.findSystemFonts()
    for f in font_list:
        try:
            name = matplotlib.font_manager.FontProperties(fname=f).get_name()
            if any(k in name.lower() for k in ["noto", "cjk", "chinese", "wqy", "droid"]):
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
        except Exception:
            pass
    matplotlib.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

setup_fonts()


# ─────────────────────────────────────────────────────────────────────
# 图1：准确率 / F1 / 延迟 综合对比（真实数据）
# ─────────────────────────────────────────────────────────────────────
def plot_accuracy_latency_comparison(summary: dict, out_dir: str):
    results = summary["results"]
    latency = summary["latency"]

    keys = ["1-LR", "2-CNN", "3-Teacher", "4-DistillCNN", "5-Features+LR"]
    names = [KEY_MAP[k][0] for k in keys]
    colors = [KEY_MAP[k][1] for k in keys]
    accs  = [results[k]["accuracy"] * 100 for k in keys]
    f1s   = [results[k]["macro_f1"] * 100 for k in keys]
    p50s  = [latency[k]["p50_ms"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Accuracy, Macro-F1 and Inference Latency Comparison\n"
        "(All results from the same experiment, same data split)",
        fontsize=13, fontweight="bold", y=1.01
    )

    # 1) 准确率
    bars = axes[0].bar(names, accs, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    axes[0].set_title("Test Accuracy (%)")
    axes[0].set_ylim(50, 100)
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")

    # 2) Macro-F1
    bars = axes[1].bar(names, f1s, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    axes[1].set_title("Macro-F1 (%)")
    axes[1].set_ylim(40, 100)
    for bar, v in zip(bars, f1s):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    # 3) 延迟（log scale）
    bars = axes[2].bar(names, p50s, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    axes[2].set_title("Inference Latency p50 (ms)")
    axes[2].set_yscale("log")
    for bar, v in zip(bars, p50s):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[2].axhline(y=0.3, color="#16a34a", linestyle="--", linewidth=1.2, alpha=0.6, label="Real-time threshold (0.3ms)")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig1_accuracy_latency.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig1_accuracy_latency.png")


# ─────────────────────────────────────────────────────────────────────
# 图2：混淆矩阵（从模型权重真实推理）
# ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(summary: dict, out_dir: str):
    import torch
    import pandas as pd
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_experiment import EnhancedCNN

    # ── 用与主实验完全一致的预处理 artifacts 重建测试集 ──
    art_path = os.path.join(out_dir, "preprocessing_artifacts.json")
    if not os.path.exists(art_path):
        print("  ✗ preprocessing_artifacts.json not found, skipping confusion matrix")
        return
    with open(art_path) as f:
        art = json.load(f)

    scaler_mean  = np.array(art["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(art["scaler_scale"], dtype=np.float32)
    feat_names   = art["feat_names"]
    label_names  = art["label_encoder_classes"]

    # 加载原始数据，应用与 load_data 完全相同的变换
    df = pd.read_pickle(summary["data"])
    label_col = "label"
    if label_col not in df.columns:
        for c in df.columns:
            if "label" in c.lower() or "class" in c.lower():
                label_col = c; break
    x = df.drop(columns=[label_col])
    y_raw = df[label_col].astype(str)
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = pd.to_numeric(x[c], errors="coerce")
    skewed_cols = [
        "duration","min_fiat","min_biat","max_fiat","max_biat",
        "mean_fiat","mean_biat","flowPktsPerSecond","flowBytesPerSecond",
        "min_flowiat","max_flowiat","mean_flowiat",
    ]
    for c in skewed_cols:
        if c in x.columns:
            x[c] = np.log1p(x[c].clip(lower=0))
    x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median())

    # 用完全相同的 test split
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    _, x_te, _, y_te = train_test_split(
        x.values, y, test_size=0.2, random_state=42, stratify=y)

    # 用 artifacts 中的 scaler 参数做变换
    x_te = (x_te - scaler_mean) / scaler_scale
    x_te = x_te.astype(np.float32)

    # ── 推理 ──
    ckpt = os.path.join(out_dir, "distill_cnn_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_c = summary["num_classes"]

    if os.path.exists(ckpt):
        model = EnhancedCNN(21, n_c).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_te).to(device)).cpu().numpy()
        y_pred = logits.argmax(axis=1)
    else:
        print("  ✗ distill_cnn_best.pt not found, skipping confusion matrix")
        return

    cm_display = confusion_matrix(y_te, y_pred)
    cm_norm = cm_display.astype(float) / cm_display.sum(axis=1, keepdims=True)

    # 验证：cm_display.sum() == n_test
    n_test_actual = cm_display.sum()
    reported_acc = summary["results"]["4-DistillCNN"]["accuracy"]
    reconstructed_acc = float((y_pred == y_te).sum()) / n_test_actual  # should equal reported_acc
    print(f"  CM test samples: {n_test_actual}  reported n_test: {summary['n_test']}")
    print(f"  CM-derived accuracy: {reconstructed_acc:.4f}  reported: {reported_acc:.4f}")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_c)); ax.set_yticks(range(n_c))
    ax.set_xticklabels(label_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(
        f"Confusion Matrix: Distilled CNN\n"
        f"Test Set, N={n_test_actual}  Accuracy={reported_acc*100:.2f}%",
        fontsize=12, fontweight="bold"
    )
    for i in range(n_c):
        for j in range(n_c):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({cm_display[i, j]})",
                    ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.85, label="Recall (row-normalized)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig2_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig2_confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────
# 图3：LR 系数热力图（可解释性核心图）
#  展示了 CNN penultimate 层（256维）上每类流量的 LR 线性可解释边界
# ─────────────────────────────────────────────────────────────────────
def plot_feature_heatmap(coef_path: str, out_dir: str):
    with open(coef_path) as f:
        data = json.load(f)

    coef = np.array(data["coefficients"])   # (7, 256)
    label_names = data["label_names"]

    # 按平均绝对值排序特征维度（最重要的放上面）
    feat_importance = np.abs(coef).mean(axis=0)
    sort_idx = np.argsort(feat_importance)[::-1]
    coef_sorted = coef[:, sort_idx]

    # 特征名重命名为 dim_0, dim_1, ... 标注 top-10
    feat_names_sorted = [f"dim_{sort_idx[i]}" for i in range(len(sort_idx))]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    vmax = np.abs(coef_sorted).max()
    im = ax.imshow(coef_sorted, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    # x轴：每10个特征显示一个标签
    step = 20
    xtick_pos = list(range(0, len(sort_idx), step))
    xtick_labels = [f"dim_{sort_idx[i]}" for i in xtick_pos]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=35, ha="right", fontsize=7)

    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("CNN Feature Dimension (sorted by importance →)", fontsize=11)
    ax.set_ylabel("Traffic Type", fontsize=11)
    ax.set_title(
        "Feature Importance Heatmap: LR Coefficients on CNN Penultimate Layer\n"
        "Red = positive contribution, Blue = negative contribution (256-dim CNN features → 7 classes)",
        fontsize=12, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="LR Coefficient")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig3_feature_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig3_feature_heatmap.png")


# ─────────────────────────────────────────────────────────────────────
# 图3b：原始 21 维流特征的 LR 系数热力图（人类可读解释）
#  每类流量的 LR 系数表示该类流量对哪些原始流特征最敏感
# ─────────────────────────────────────────────────────────────────────
def plot_raw_feature_importance(summary: dict, out_dir: str):
    import os
    raw_path = os.path.join(out_dir, "lr_raw_coefficients.json")
    if not os.path.exists(raw_path):
        print("  WARNING: lr_raw_coefficients.json not found, skipping raw feature plot")
        return
    with open(raw_path) as f:
        raw_data = json.load(f)

    label_names = raw_data["label_names"]
    feat_names  = raw_data["feat_names"]   # 21 original flow features
    coef = np.array(raw_data["coefficients"])   # (7, 21)

    # 按平均绝对值排序
    imp = np.abs(coef).mean(axis=0)
    sort_idx = np.argsort(imp)[::-1]
    coef_sorted = coef[:, sort_idx]
    feat_sorted = [feat_names[i] for i in sort_idx]

    vmax = np.abs(coef_sorted).max()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # 左：热力图（按重要性排序的特征）
    ax = axes[0]
    im = ax.imshow(coef_sorted, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(feat_sorted)))
    ax.set_xticklabels(feat_sorted, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_title(
        "LR Coefficients on 21 Raw Flow Features\n"
        "(Red=positive, Blue=negative contribution, sorted by importance)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Flow Feature (sorted by importance ->)", fontsize=10)
    ax.set_ylabel("Traffic Type", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="LR Coefficient")

    # 右：per-class bar chart — 每个流量类型的 top-5 特征
    ax = axes[1]
    n_top = 5
    bar_colors = plt.cm.tab10(np.linspace(0, 0.9, len(label_names)))
    bar_w = 0.12
    for i, (cls, color) in enumerate(zip(label_names, bar_colors)):
        coeffs = coef[i]
        top_idx = np.argsort(np.abs(coeffs))[::-1][:n_top]
        top_feats = [feat_names[j] for j in top_idx]
        top_vals = [coeffs[j] for j in top_idx]
        x_pos = np.arange(n_top) + i * bar_w
        bars = ax.bar(x_pos, top_vals, width=bar_w, label=cls, color=color, alpha=0.85)
        for bar, val in zip(bars, top_vals):
            color2 = "white" if abs(val) > max(np.abs(top_vals))*0.6 else "black"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(top_vals),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=5.5, color=color2)
    ax.set_xticks(np.arange(n_top) + bar_w * (len(label_names)-1) / 2)
    ax.set_xticklabels([f"Top-{k}" for k in range(1, n_top+1)], fontsize=9)
    ax.set_ylabel("LR Coefficient", fontsize=10)
    ax.set_title("Top-5 Discriminative Flow Features per Traffic Type\n(LR Coefficients on 21 Raw Features)",
                 fontsize=11, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_dir + "/fig3b_raw_feature_importance.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  OK fig3b_raw_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────
# 图4：t-SNE 特征空间可视化（真实特征，非模拟）
# ─────────────────────────────────────────────────────────────────────
def plot_tsne(summary: dict, out_dir: str):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = pd.read_pickle(summary["data"])
    label_col = "label"
    if label_col not in df.columns:
        for c in df.columns:
            if "label" in c.lower() or "class" in c.lower():
                label_col = c; break
    x = df.drop(columns=[label_col]); y_raw = df[label_col].astype(str)
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = pd.to_numeric(x[c], errors="coerce")
    skewed_cols = ["duration","min_fiat","min_biat","max_fiat","max_biat",
                   "mean_fiat","mean_biat","flowPktsPerSecond","flowBytesPerSecond",
                   "min_flowiat","max_flowiat","mean_flowiat"]
    for c in skewed_cols:
        if c in x.columns:
            x[c] = np.log1p(x[c].clip(lower=0))
    x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median())
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder(); y = le.fit_transform(y_raw)
    label_names = list(le.classes_)
    _, x_te, _, y_te = train_test_split(x.values, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    x_te = scaler.fit_transform(x_te)

    # 子采样加速
    n_sub = min(2000, len(x_te))
    idx = np.random.choice(len(x_te), n_sub, replace=False)
    x_sub, y_sub = x_te[idx], y_te[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init="pca")
    x_emb = tsne.fit_transform(x_sub)

    cmap = plt.cm.get_cmap("tab10", len(label_names))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for i, name in enumerate(label_names):
        mask = y_sub == i
        ax.scatter(x_emb[mask, 0], x_emb[mask, 1], c=[cmap(i)], label=name, s=15, alpha=0.65)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title(
        "t-SNE Visualization of Test Set Features (21-dim → 2-dim)\n"
        "Each dot = one traffic flow, color = traffic type",
        fontsize=12, fontweight="bold"
    )
    ax.legend(title="Traffic Type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4_tsne.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig4_tsne.png")


# ─────────────────────────────────────────────────────────────────────
# 图5：收敛曲线（验证集，不是测试集）
#  训练过程中只在验证集上评估，避免测试集泄漏
# ─────────────────────────────────────────────────────────────────────
def plot_convergence(summary: dict, out_dir: str):
    distill_hist = summary.get("distill_history", {})
    teacher_hist = summary.get("teacher_history", {})

    distill_val_acc = distill_hist.get("val_acc", [])
    distill_val_f1  = distill_hist.get("val_f1", [])
    distill_loss    = distill_hist.get("train_loss", [])
    teacher_val_acc = teacher_hist.get("val_acc", [])
    teacher_val_f1  = teacher_hist.get("val_f1", [])

    distill_epochs = range(1, len(distill_val_acc) + 1)
    teacher_epochs = range(1, len(teacher_val_acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Training Convergence (Validation Set, No Test Leakage)\n"
        "Dashed line = final test set performance reported in Table",
        fontsize=12, fontweight="bold"
    )

    # 左：蒸馏 CNN 收敛
    ax = axes[0]
    color = COLORS["Distill CNN"]
    ax.plot(distill_epochs, distill_loss, color=color, linewidth=2, label="Train Loss (蒸馏)")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Training Loss", fontsize=10, color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax2 = ax.twinx()
    ax2.plot(distill_epochs, [v*100 for v in distill_val_acc], color="#2563eb",
             linewidth=2, linestyle="--", label="Val Accuracy")
    ax2.plot(distill_epochs, [v*100 for v in distill_val_f1], color="#f97316",
             linewidth=2, linestyle=":", label="Val Macro-F1")
    ax2.set_ylabel("Validation Accuracy / F1 (%)", fontsize=10)
    ax2.tick_params(axis="y")
    ax.set_title(f"Distilled CNN (150 epochs)\nVal Acc: {distill_val_acc[-1]*100:.1f}%  Val F1: {distill_val_f1[-1]*100:.1f}%")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(1, len(distill_epochs))

    # 右：教师 Transformer 收敛
    ax = axes[1]
    ax.plot(teacher_epochs, [v*100 for v in teacher_val_acc], color=COLORS["Teacher"],
             linewidth=2, label="Val Accuracy")
    ax.plot(teacher_epochs, [v*100 for v in teacher_val_f1], color="#f97316",
             linewidth=2, linestyle="--", label="Val Macro-F1")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation Accuracy / F1 (%)", fontsize=10)
    ax.set_title(f"Transformer Teacher (200 epochs)\nVal Acc: {teacher_val_acc[-1]*100:.1f}%  Val F1: {teacher_val_f1[-1]*100:.1f}%")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, len(teacher_epochs))

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig5_convergence.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ fig5_convergence.png")



# ─────────────────────────────────────────────────────────────────────
# 图6：消融实验（T 和 α）
#  与主实验完全一致的数据分割和训练设置：
#  - 相同的 train/val split（seed=42, val_size=0.15, test_size=0.2）
#  - 相同的 150 epochs
#  - 相同的 label_smoothing=0.1
#  - 相同的 seed=42（模型初始化）
#  唯一变量：temperature T 和 distillation loss weight alpha


def plot_ablation(summary: dict, out_dir: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score
    from sklearn.utils.class_weight import compute_class_weight
    import pandas as pd
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_experiment import EnhancedCNN, TabTransformer, load_data

    data_path = summary["data"]
    num_classes = summary["num_classes"]
    base_T = float(summary["temperature"])
    base_alpha = float(summary["alpha"])
    label_smoothing = float(summary.get("label_smoothing", 0.1))

    print("  Reconstructing train/val split identically to main experiment...")
    x_tr, x_va, x_te, y_tr, y_va, y_te, le, scaler, feat_names = load_data(
        data_path, seed=42, val_size=0.15, test_size=0.2
    )
    label_names = list(le.classes_)
    print(f"  Train: {len(y_tr)}  Val: {len(y_va)}  Test: {len(y_te)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_ckpt = os.path.join(out_dir, "teacher_best.pt")
    if not os.path.exists(teacher_ckpt):
        print("  ERROR: teacher_best.pt not found, skipping ablation")
        return
    teacher = TabTransformer(
        21, num_classes,
        d_model=summary.get("d_model", 256),
        n_heads=summary.get("n_heads", 8),
        n_layers=summary.get("n_layers", 6),
    ).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    teacher_dev = next(teacher.parameters()).device
    print(f"  Loaded saved teacher from {teacher_ckpt} (device={teacher_dev})")

    dl_tr = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long()),
        batch_size=128, shuffle=True)
    dl_va = DataLoader(
        TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va).long()),
        batch_size=2048, shuffle=False)

    def train_and_eval(T_val, alpha_val, seed=42):
        """
        完全复制主实验 train_cnn_distill 的训练逻辑。
        唯一差异：temperature 和 alpha 由参数传入。
        """
        torch.manual_seed(seed)
        model = EnhancedCNN(21, num_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
        loss_fn_kl = nn.KLDivLoss(reduction="batchmean")
        class_weights_arr = compute_class_weight(
            "balanced", classes=np.arange(num_classes), y=y_tr
        )
        class_weights = torch.from_numpy(class_weights_arr).float().to(device)
        criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        best_val_acc = 0.0
        for ep in range(1, 151):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                with torch.no_grad():
                    soft = F.softmax(teacher(xb.to(teacher_dev)) / T_val, dim=1)
                soft_loss = loss_fn_kl(F.log_softmax(logits / T_val, dim=1), soft)
                hard_loss = criterion_ce(logits, yb)
                loss = alpha_val * soft_loss * T_val**2 + (1 - alpha_val) * hard_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            scheduler.step()

            # 在验证集上评估（与主实验一致的 best-model 保存策略）
            model.eval()
            with torch.no_grad():
                logits_va = model(torch.from_numpy(x_va).to(device)).cpu().numpy()
            val_acc = float((logits_va.argmax(axis=1) == y_va).mean())
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if ep % 50 == 0 or ep == 150:
                print(f"      T={T_val} α={alpha_val} ep={ep:3d}/150  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

        return best_val_acc

    Ts = [2, 3, 4, 5, 6]
    alphas = [0.3, 0.5, 0.7, 0.9]
    acc_matrix = np.zeros((len(alphas), len(Ts)))

    print(f"\n  Running ablation ({len(Ts)*len(alphas)} configs, 150 epochs each, seed=42)...")
    print(f"  Settings: label_smoothing={label_smoothing}, batch_size=128, lr=5e-4, T_max=150")
    for i, a in enumerate(alphas):
        for j, t in enumerate(Ts):
            acc_matrix[i, j] = train_and_eval(t, a)
            print(f"    T={t}, α={a} -> best_val_acc={acc_matrix[i,j]*100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Ablation Study: Temperature T and Distillation Loss Weight α\n"
        "(Same train/val split, 150 epochs, seed=42 as main experiment)",
        fontsize=13, fontweight="bold"
    )
    base_j = Ts.index(base_T)
    base_i = alphas.index(base_alpha)
    main_test_acc = summary["results"]["4-DistillCNN"]["accuracy"] * 100
    main_val_acc = summary["distill_history"]["val_acc"][-1] * 100

    ax = axes[0]
    cmap_T = plt.cm.viridis
    for i, a in enumerate(alphas):
        label = f"α={a}" + (" [selected]" if a == base_alpha else "")
        ax.plot(Ts, acc_matrix[i] * 100, marker="o", linewidth=2.5, markersize=7,
                color=cmap_T(i / max(len(alphas) - 1, 1)), label=label)
    ax.axvline(x=base_T, color="gray", linestyle="--", alpha=0.7)
    ax.scatter([base_T], [acc_matrix[base_i][base_j] * 100], s=200, zorder=10,
               color="red", marker="*", label=f"Selected (T={base_T})")
    ax.axhline(y=main_test_acc, color="navy", linestyle=":", linewidth=1.5,
               label=f"Main test acc ({main_test_acc:.1f}%)")
    ax.set_xlabel("Temperature T", fontsize=11)
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=11)
    ax.set_title(
        f"Temperature Sensitivity (α={base_alpha})\n"
        f"Selected T={base_T}: {acc_matrix[base_i][base_j]*100:.2f}% (val)",
        fontsize=10
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(80, 96)

    ax = axes[1]
    cmap_a = plt.cm.plasma
    for j, t in enumerate(Ts):
        label = f"T={t}" + (" [selected]" if t == base_T else "")
        ax.plot(alphas, acc_matrix[:, j] * 100, marker="s", linewidth=2.5, markersize=7,
                color=cmap_a(j / max(len(Ts) - 1, 1)), label=label)
    ax.axvline(x=base_alpha, color="gray", linestyle="--", alpha=0.7)
    ax.scatter([base_alpha], [acc_matrix[base_i][base_j] * 100], s=200, zorder=10,
               color="red", marker="*", label=f"Selected (α={base_alpha})")
    ax.axhline(y=main_test_acc, color="navy", linestyle=":", linewidth=1.5,
               label=f"Main test acc ({main_test_acc:.1f}%)")
    ax.set_xlabel("Distillation Loss Weight α", fontsize=11)
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=11)
    ax.set_title(
        f"α Sensitivity (T={base_T})\n"
        f"Selected α={base_alpha}: {acc_matrix[base_i][base_j]*100:.2f}% (val)",
        fontsize=10
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(80, 96)

    plt.tight_layout()
    plt.savefig(out_dir + "/fig6_ablation.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ fig6_ablation.png  (selected: T={base_T}, α={base_alpha}, val={acc_matrix[base_i][base_j]*100:.2f}%)")

    # 保存原始数据供检查
    import json as _json
    acc_data = {
        "Ts": Ts, "alphas": alphas,
        "acc_matrix": acc_matrix.tolist(),
        "selected_T": base_T, "selected_alpha": base_alpha,
        "selected_val_acc": float(acc_matrix[base_i][base_j]),
        "main_test_acc": main_test_acc,
    }
    with open(os.path.join(out_dir, "ablation_data.json"), "w") as f:
        _json.dump(acc_data, f, indent=2)
    print("  ✓ ablation_data.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="Path to experiment run directory")
    ap.add_argument("--skip_tsne",    action="store_true", help="Skip t-SNE")
    ap.add_argument("--skip_ablation", action="store_true", help="Skip ablation study (~5 min)")
    args = ap.parse_args()

    if args.run_dir is None:
        import glob, os
        runs = sorted(glob.glob("runs/experiment_*"), key=os.path.getmtime, reverse=True)
        if not runs:
            print("Error: no experiment runs found"); return
        args.run_dir = runs[0]
        print(f"Auto-selected latest run: {args.run_dir}")

    out_dir = args.run_dir
    with open(f"{out_dir}/summary.json") as f:
        summary = json.load(f)

    print(f"\nLoading results from: {out_dir}")
    for k, v in summary["results"].items():
        print(f"  {k}: acc={v['accuracy']:.4f}  f1={v['macro_f1']:.4f}")

    print("\nGenerating figures...")
    plot_accuracy_latency_comparison(summary, out_dir)
    plot_confusion_matrix(summary, out_dir)
    plot_feature_heatmap(f"{out_dir}/lr_coefficients.json", out_dir)
    plot_raw_feature_importance(summary, out_dir)
    if not args.skip_tsne:
        plot_tsne(summary, out_dir)
    plot_convergence(summary, out_dir)
    if not args.skip_ablation:
        plot_ablation(summary, out_dir)

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
