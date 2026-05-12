import matplotlib.pyplot as plt
import numpy as np
import torch
from Pruning.pruning_CSSP import *

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

    layers = layers[:-1]

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
        "pruning_filter_l1": "C3",
        "pruning_filter_l2": "C4",
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



def plot_singular_values(pruned_models_dict, model_baseline, params_base, X, l, rho_key,
                                     methods=None, device=None, log_scale=True,
                                     normalize=True, title=None):
    """
    rho_key: the key to index pruned_models_dict[method], e.g., "0.95" or "0.90"
    pruned_models_dict[method][rho_key] = pruned_model
    layer_l: index in params, not global layer index
    """

    if device is None:
        try:
            device = next(model_baseline.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    X = X.to(device)
    plt.figure(figsize=(8, 5), dpi=200)

    singular_values_dict = {}

    # baseline
    model_baseline = model_baseline.to(device).eval()
    A_base = get_layer_activation_matrix(model_baseline, X, params_base, l)
    print(A_base.shape)
    s_base = compute_singular_values(A_base)
    if normalize: 
        s_base = s_base / (s_base[0] + 1e-12)

    singular_values_dict["baseline"] = s_base
    plt.plot(np.arange(1, len(s_base) + 1), s_base, linewidth=1.2,
             linestyle="--", label="baseline")

    # pruned models
    for method in methods:
        if method not in pruned_models_dict:
            print(f"[Skip] method {method} not found.")
            continue

        pruned_models = pruned_models_dict[method]

        if rho_key not in pruned_models:
            print(f"[Skip] rho_key {rho_key} not found for method {method}. "
                  f"Available keys: {list(pruned_models.keys())}")
            continue

        model = pruned_models[rho_key].to(device).eval()
        params = extract_params(model.model)
        A = get_layer_activation_matrix(model, X, params, l)
        s = compute_singular_values(A)
        if normalize: s = s / (s[0] + 1e-12)

        singular_values_dict[method] = s
        plt.plot(np.arange(1, len(s) + 1), s, marker="o",
                 markersize=0.8, linewidth=0.6, label=method)

    if log_scale: plt.yscale("log")

    plt.xlabel("Singular value index")
    plt.ylabel(r"Relative Singular Value ($\sigma_i / \sigma_1$)") if normalize else plt.ylabel("Singular Value")
    plt.title(title or f"Singular Value Spectrum at Layer {params_base[l]['layer_idx']}, Ratio = {rho_key}")
    #plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

    return singular_values_dict







# Plot per-sample argmax index consistency for the last activation matrix
def argmax_consistency(A_base, A_pruned):
    A_base = A_base.detach().cpu()
    A_pruned = A_pruned.detach().cpu()

    pred_base = A_base.argmax(dim=1)
    pred_pruned = A_pruned.argmax(dim=1)

    return (pred_base == pred_pruned).float().mean().item()


def plot_argmax_consistency(
    activation_matrices,
    baseline_key="baseline",
    methods=None,
    title=None,
):
    """
    activation_matrices[method] = activation matrix
    """

    if methods is None:
        methods = [m for m in activation_matrices.keys() if m != baseline_key]

    A_base = activation_matrices[baseline_key]

    results = {}

    for method in methods:
        if method not in activation_matrices:
            print(f"[Skip] {method} not found.")
            results[method] = np.nan
            continue

        results[method] = argmax_consistency(A_base, activation_matrices[method])

    plt.figure(figsize=(6, 3), dpi=200)
    plt.bar(list(results.keys()), list(results.values()))

    plt.ylabel("Argmax consistency")
    plt.ylim(0, 1.02)
    plt.title(title or "Final-layer Argmax Consistency")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return results











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




def plot_structured_vs_magnitude_heatmaps(layerwise_results, ratio_key, structured_methods=("StrongRRQR", "RPCholesky", "ARP"), magnitude_method="magnitude_pruning"):
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

    structured_data = {m: layerwise_results[m][ratio_key] for m in structured_methods}
    magnitude_data = {magnitude_method: layerwise_results[magnitude_method][ratio_key]}

    structured_layers = set().union(*[set(d.keys()) for d in structured_data.values()])
    magnitude_layers = set(magnitude_data[magnitude_method].keys())
    common_layers = sorted(structured_layers & magnitude_layers)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=200, gridspec_kw={"height_ratios": [3, 1]})

    for ax, heatmap_data, title in zip(axes, [structured_data, magnitude_data], [f"CSSP: channel / neuron retention at params ratio = {ratio_key}", f"Magnitude pruning: nonzero-weight retention at params ratio = {ratio_key}"]):
        methods = list(heatmap_data.keys())
        layers = common_layers
        data = np.array([[heatmap_data[method].get(layer, np.nan) for layer in layers] for method in methods])

        im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="viridis")

        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods, fontweight="bold")
        ax.set_xlabel("Layer Index")
        ax.set_title(title)

    fig.subplots_adjust(right=0.88, hspace=0.75)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.015, 0.5]) 
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


