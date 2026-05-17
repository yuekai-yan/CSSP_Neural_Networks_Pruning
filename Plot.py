import matplotlib.pyplot as plt
import numpy as np
import torch
from Pruning.pruning_CSSP import *

# For pruning curve
def plot_pruning_curve(base_acc, ratios, accs, labels, crit, ylabel,
                       title=None,
                       base_loss=None, losses=None):
    two_plots = base_loss is not None and losses is not None

    if two_plots:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(4.8, 3.0), dpi=300)
        axes = [axes]

    def draw_one(ax, base, ys, ylab, ttl):
        for y, label in zip(ys, labels):
            ax.plot(
                ratios, y,
                marker="o",
                markersize=3.0,
                linewidth=1.25,
                label=label,
                alpha=0.95,
            )

        ax.axhline(
            base,
            linestyle=(0, (4, 2)),
            linewidth=1.4,
            color="0.25",
            label="Baseline",
            zorder=0,
        )

        ax.annotate(
            f"{base:.2f}",
            xy=(0, base),
            xycoords=ax.get_yaxis_transform(),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.5,          
            color="black",       
        )   

        xlabel = {
            "flops": "Fraction of FLOPs Remaining",
            "params": "Fraction of Parameters Remaining",
        }.get(crit, crit)

        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)

        if ttl is not None:
            ax.set_title(ttl, fontsize=9.5, pad=5)

        ax.tick_params(axis="both", labelsize=7.5, width=0.7, length=3)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.28)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)

        ax.margins(x=0.02, y=0.06)

    draw_one(axes[0], base_acc, accs, ylabel, title)

    if two_plots:
        draw_one(axes[1], base_loss, losses, "Test Loss", None)

        handles, legend_labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=3,
            fontsize=7.5,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.2,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig("pruning_curve_acc_loss.pdf", bbox_inches="tight")
        plt.savefig("pruning_curve_acc_loss.png", dpi=300, bbox_inches="tight")

    else:
        axes[0].legend(
            fontsize=7,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            handlelength=1.8,
            columnspacing=1.0,
        )

        fig.tight_layout()
        plt.savefig("pruning_curve.pdf", bbox_inches="tight")
        plt.savefig("pruning_curve.png", dpi=300, bbox_inches="tight")

    plt.show()


# For layerwise retention heatmap
def plot_layerwise_retention_heatmap(heatmap_data, title=None):
    methods = list(heatmap_data.keys())

    layers = sorted({
        layer
        for method in methods
        for layer in heatmap_data[method].keys()
    })
    layers = layers[:-1]

    data = np.array([
        [heatmap_data[method].get(layer, np.nan) for layer in layers]
        for method in methods
    ])

    fig, ax = plt.subplots(figsize=(8.0, 2.6), dpi=400)

    im = ax.imshow(
        data,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="YlGnBu_r",          
        interpolation="nearest",
    )

    for i in range(len(methods)):
        for j in range(len(layers)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="black",
                    fontweight="normal",
                )

    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)

    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods, fontsize=7.5)

    ax.set_xlabel("Layer Index", fontsize=8)

    if title:
        ax.set_title(title, fontsize=9.5, pad=6)

    ax.tick_params(axis="both", width=0.6, length=2.5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.022,
        pad=0.018,
    )
    cbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout()
    plt.show()

# For singular value
def plot_singular_values(
    pruned_models_dict,
    model_baseline,
    params_base,
    X,
    layer_ls,
    rho_key,
    methods,
    device=None,
    log_scale=True,
    normalize=True,
    title=None,
):
    if device is None:
        try:
            device = next(model_baseline.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    X = X.to(device)
    model_baseline = model_baseline.to(device).eval()

    fig, axes = plt.subplots(2, 4, figsize=(12.8, 4.8), dpi=300)
    axes = axes.ravel()


    singular_values_dict = {}

    for i, (ax, l) in enumerate(zip(axes, layer_ls)):
        singular_values_dict[l] = {}

        # baseline
        A_base = get_layer_activation_matrix(model_baseline, X, params_base, l)
        s_base = compute_singular_values(A_base)

        if normalize:
            s_base = s_base / (s_base[0] + 1e-12)

        singular_values_dict[l]["Baseline"] = s_base

        for method in methods:
            if method not in pruned_models_dict:
                print(f"[Skip] method {method} not found.")
                continue

            if rho_key not in pruned_models_dict[method]:
                print(f"[Skip] rho_key {rho_key} not found for method {method}.")
                continue

            model = pruned_models_dict[method][rho_key].to(device).eval()
            params = extract_params(model.model)

            A = get_layer_activation_matrix(model, X, params, l)
            s = compute_singular_values(A)

            if normalize:
                s = s / (s[0] + 1e-12)

            singular_values_dict[l][method] = s

            ax.plot(
                np.arange(1, len(s) + 1),
                s,
                marker="o",
                markersize=1.0,
                linewidth=0.8,
                label=method,
                alpha=0.95,
            )

        ax.plot(
            np.arange(1, len(s_base) + 1),
            s_base,
            linewidth=1.2,
            linestyle=(0, (4, 2)),
            color="0.25",
            label="Baseline",
            zorder=0,
        )

        if log_scale:
            ax.set_yscale("log")

        if i >= 4:
            ax.set_xlabel("Singular value index", fontsize=8.5)
        else:
            ax.set_xlabel("")

        if ax in axes[::4]:
            ax.set_ylabel(
                r"$\sigma_i / \sigma_1$" if normalize else "Singular Value",
                fontsize=8.5,
            )
        else:
            ax.set_ylabel("")

        ax.set_title(
            f"Layer {params_base[l]['layer_idx']}",
            fontsize=9.5,
            pad=5,
        )

        ax.tick_params(axis="both", labelsize=7.5, width=0.7, length=3)
        ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.28)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)

    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        fontsize=7.5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )

    if title:
        fig.suptitle(title, fontsize=10, y=1.08)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


