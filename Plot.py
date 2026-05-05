import matplotlib.pyplot as plt
import numpy as np
import torch

# For pruning curve
def plot_pruning_curve(base_acc, ratios, accs, labels, crit, ylabel, title=None):
    plt.figure(figsize=(5.5, 3), dpi=150)

    for i in range(len(accs)):
        linestyle = "--" if labels[i] == "pruning_filter" else "-"

        plt.plot(
            ratios,
            accs[i],
            marker="o",
            markersize=4,
            linewidth=1.5,
            linestyle=linestyle,
            label=labels[i],
        )

    plt.axhline(
        base_acc,
        linestyle="--",
        linewidth=1.5,
        label="Baseline",
    )

    if crit == "flops":
        plt.xlabel("Fraction of FLOPs Remaining", fontsize=8, labelpad=2)
    elif crit == "params":
        plt.xlabel("Fraction of Parameters Remaining", fontsize=8, labelpad=2)
    plt.ylabel(ylabel, fontsize=8, labelpad=2)

    plt.title(title, fontsize=15, fontweight="bold", pad=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    #plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=8, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(left=0.12, bottom=0.13)
    plt.show()


# For layerwise retention heatmap
def plot_layerwise_retention_heatmap(heatmap_data, title=None):

    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

    methods = list(heatmap_data.keys())

    layers = sorted({
        layer
        for method in methods
        for layer in heatmap_data[method].keys()
    })

    data = np.array([
        [heatmap_data[method].get(layer, np.nan) for layer in layers]
        for method in methods
    ])

    plt.figure(figsize=(10, 4), dpi=200)
    im = plt.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="viridis")

    plt.colorbar(im)
    plt.xticks(np.arange(len(layers)), layers)
    plt.yticks(np.arange(len(methods)), methods, fontweight="bold")
    #plt.yticks(np.arange(len(methods)), methods, rotation=30, ha="right")

    plt.xlabel("Layer Index")
    #plt.ylabel("Method")

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()


# For singular value spectrum
def plot_singular_value_spectrum(s, normalize=True, log_scale=True, title=None):
    if normalize:
        s_plot = s / s[0]
        ylabel = "Normalized Singular Value"
    else:
        s_plot = s
        ylabel = "Singular Value"

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(s_plot) + 1), s_plot, marker="o", markersize=3, linewidth=0.8)

    if log_scale:
        plt.yscale("log")

    #plt.xlabel("Index")
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    #plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_singular_value_spectrum(
    s,
    A=None,
    normalize=True,
    log_scale=True,
    title=None,
    heatmap_data=None,
    methods=None,
    layer_idx=None,
):
    plt.rcParams["font.family"] = "Arial"      
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    if normalize:
        s_plot = s / s[0]
        ylabel = r"Relative Singular Value ($\sigma_i / \sigma_1$)"
    else:
        s_plot = s
        ylabel = "Singular Value"

    if torch.is_tensor(s_plot):
        s_plot = s_plot.detach().cpu().numpy()
    else:
        s_plot = np.asarray(s_plot)

    x = np.arange(1, len(s_plot) + 1)

    plt.figure(figsize=(7, 4.5), dpi=200)

    plt.plot(
        x,
        s_plot,
        #color="black",
        marker="o",
        markersize=2,
        linewidth=0.5,
        label="Spectrum",
    )

    if log_scale:
        plt.yscale("log")

    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    method_colors = {
        "StrongRRQR": "C0",
        "RPCholesky": "C1",
        "ARP": "C2",
        "pruning_filter": "C3",
    }  

    if log_scale:
        y_min, y_max = np.min(s_plot[s_plot > 0]), np.max(s_plot)
        y_text_global = 10 ** (np.log10(y_min) * 0.15 + np.log10(y_max) * 0.85)
    else:
        y_min, y_max = np.min(s_plot), np.max(s_plot)
        y_text_global = y_min + 0.85 * (y_max - y_min)

    if heatmap_data is not None and methods is not None:
        n_cols = A.shape[1] if A is not None else len(s_plot)

        for method in methods:
            value = heatmap_data[method]

            # value can be either:
            # 1. a float ratio
            # 2. a dict: {layer_idx: ratio}
            if isinstance(value, dict):
                if layer_idx is None:
                    raise ValueError("layer_idx must be provided when heatmap_data[method] is a dict.")

                ratio = value[layer_idx]
            else:
                ratio = value

            x_pos = int(ratio * n_cols)
            x_pos = max(1, min(x_pos, len(s_plot)))
            y_pos = s_plot[x_pos - 1]

            color = method_colors.get(method, None)

            plt.axvline(
                x=x_pos,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
            y_text = y_text_global

            plt.text(
                x_pos,
                y_text,
                method,
                color=color,
                fontsize=8,
                fontweight="bold",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
            )
            plt.scatter(x_pos, y_pos, color=color, s=18, zorder=5)
            plt.text(
                x_pos + 0.5,
                y_pos * 1.15,
                f"{y_pos:.2e}",
                color=color, 
                fontsize=7,
                fontweight="bold",
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
            )

    plt.tight_layout()
    plt.show()


# overall relative Frobrnius error trajectory
def plot_Frobenius_error_trajectory(recon_results):
    plt.figure(figsize=(7, 5))

    for method, hist in recon_results.items():
        y = [item["reconstruction_error"] for item in hist]
        x = list(range(1, len(y) + 1))

        plt.plot(x, y, marker="o", label=method)

    plt.xlabel("Pruning Step")
    plt.ylabel(r"Relative Frobenius Error $\frac{\|A-A(:, J)A(:, J)^{\dagger}A\|_F}{\|A\|_F}$")
    plt.title("Reconstruction Error Along Iterative Pruning")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_fixed_layer_relerr(relerr_results, keep_ratios, layer_idx, svd_bound=None, title=None):
    plt.figure(figsize=(5, 3), dpi=150)

    for method, errors in relerr_results.items():
        plt.plot(keep_ratios, errors, marker="o", linewidth=0.8, markersize=3, label=method)

    if svd_bound is not None:
        plt.plot(keep_ratios, svd_bound, linestyle="--", marker="s", linewidth=0.8, markersize=3, color="C0", label="SVD lower bound")

    plt.xlabel("Remaining Columns Ratio", fontsize=8)

    plt.ylabel(
        r"Relative Frobenius Error $\frac{\|A-A(:, J)A(:, J)^{\dagger}A\|_F}{\|A\|_F}$",
        fontsize=8,
        labelpad=9
    )

    plt.title(title, fontsize=10)
    plt.legend(fontsize=7)

    plt.tight_layout()
    plt.show()