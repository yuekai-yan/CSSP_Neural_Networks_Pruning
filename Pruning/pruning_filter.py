import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from Pruning.pruning_CSSP import *

def filter_pruning_l1(params, l, keep_rank):
    """
    L1-norm based filter pruning.
    Prune output channels / neurons of params[l],
    and adjust the input dimension of the next prunable layer.
    """
    layer = params[l]
    W = layer["weight"]

    if layer["layer_type"] in ["Conv2d", "ConvBNReLU"]:
        # W: [out_channels, in_channels, k, k]
        scores = W.abs().reshape(W.shape[0], -1).sum(dim=1)

    elif layer["layer_type"] in ["Linear", "LinearBNReLU"]:
        # W: [out_features, in_features]
        scores = W.abs().sum(dim=1)

    else:
        return params

    keep_idx = torch.argsort(scores, descending=True)[:keep_rank]
    keep_idx, _ = torch.sort(keep_idx)

    # prune current layer output
    if layer["layer_type"] in ["Conv2d", "ConvBNReLU"]:
        layer["weight"] = layer["weight"][keep_idx, :, :, :].clone()

        if layer.get("bias", None) is not None:
            layer["bias"] = layer["bias"][keep_idx].clone()

        if "bn_weight" in layer:
            layer["bn_weight"] = layer["bn_weight"][keep_idx].clone()
            layer["bn_bias"] = layer["bn_bias"][keep_idx].clone()
            layer["running_mean"] = layer["running_mean"][keep_idx].clone()
            layer["running_var"] = layer["running_var"][keep_idx].clone()

    elif layer["layer_type"] in ["Linear", "LinearBNReLU"]:
        layer["weight"] = layer["weight"][keep_idx, :].clone()

        if layer.get("bias", None) is not None:
            layer["bias"] = layer["bias"][keep_idx].clone()

        if "bn_weight" in layer:
            layer["bn_weight"] = layer["bn_weight"][keep_idx].clone()
            layer["bn_bias"] = layer["bn_bias"][keep_idx].clone()
            layer["running_mean"] = layer["running_mean"][keep_idx].clone()
            layer["running_var"] = layer["running_var"][keep_idx].clone()

    # adjust next layer input channels/features
    for j in range(l + 1, len(params)):
        next_layer = params[j]

        if next_layer["layer_type"] in ["Conv2d", "ConvBNReLU"]:
            next_layer["weight"] = next_layer["weight"][:, keep_idx, :, :].clone()
            break

        elif next_layer["layer_type"] in ["Linear", "LinearBNReLU"]:
            next_W = next_layer["weight"]

            # Conv -> Flatten -> Linear case
            if layer["layer_type"] in ["Conv2d", "ConvBNReLU"]:
                old_out_channels = W.shape[0]
                spatial_size = next_W.shape[1] // old_out_channels

                expanded_idx = []
                for idx in keep_idx:
                    start = int(idx) * spatial_size
                    end = start + spatial_size
                    expanded_idx.extend(range(start, end))

                expanded_idx = torch.tensor(
                    expanded_idx,
                    dtype=torch.long,
                    device=next_W.device,
                )

                next_layer["weight"] = next_W[:, expanded_idx].clone()

            # Linear -> Linear case
            else:
                next_layer["weight"] = next_W[:, keep_idx].clone()

            break

    return params

def iterative_filter_pruning(
    model0,
    X,
    input_shape,
    rho,
    step_size,
    test_loader,
    train_loader=None,
    S=None,
    device=None,
    use_bn_recalibration=True,
    bn_batches=20,
):
    """
    Iterative L1 filter pruning.

    Inputs:
        model0:      original model
        X:           sample input batch
        input_shape: input image shape, e.g. (3, 32, 32)
        rho:         target FLOPs ratio
        step_size:   per-layer keep ratio, e.g. 0.9
        S:           layer indices not to prune
        crit:        str, "flops" or "params", criterion for stopping pruning

    Output:
        model:       pruned model
    """

    if device is None:
        try:
            device = next(model0.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model = model0
    model.eval()

    if S is None:
        S = {len(model0.model) - 1}

    input_shape_origin = input_shape

    F0 = compute_total_flops(model0.model, input_shape_origin)
    F = F0

    params = extract_params(model.model)
    original_widths = {
        layer["layer_idx"]: layer["weight"].shape[0]
        for layer in params
        if layer["layer_type"] in ["Conv2d", "ConvBNReLU", "Linear", "LinearBNReLU"]
    }

    accs = []
    test_losses = []
    layerwise_history = {}

    for j in rho:
        while F > F0 * j:
            infos = []
            input_shape = input_shape_origin
            forward_matrix = X

            for i, layer in enumerate(params[:-1]):

                if layer["layer_idx"] in S or layer["layer_type"] == "Flatten":
                    forward_matrix = forward_from_a_to_b(
                        model.model,
                        forward_matrix,
                        layer["layer_idx"],
                        params[i + 1]["layer_idx"],
                    )
                    input_shape = forward_matrix.shape[1:]
                    continue

                forward_matrix = forward_from_a_to_b(
                    model.model,
                    forward_matrix,
                    layer["layer_idx"],
                    params[i + 1]["layer_idx"],
                )

                W = layer["weight"]

                if layer["layer_type"] not in [
                    "Conv2d",
                    "ConvBNReLU",
                    "Linear",
                    "LinearBNReLU",
                ]:
                    input_shape = forward_matrix.shape[1:]
                    continue

                keep_rank = int(W.shape[0] * step_size)

                if keep_rank < 1:
                    input_shape = forward_matrix.shape[1:]
                    continue

                if keep_rank >= W.shape[0]:
                    input_shape = forward_matrix.shape[1:]
                    continue

                # L1 filter importance
                if layer["layer_type"] in ["Conv2d", "ConvBNReLU"]:
                    filter_scores = W.abs().reshape(W.shape[0], -1).sum(dim=1)
                else:
                    filter_scores = W.abs().sum(dim=1)

                sorted_scores, _ = torch.sort(filter_scores, descending=False)

                num_pruned = W.shape[0] - keep_rank

                # estimated error caused by pruning low-norm filters
                err = (
                    sorted_scores[:num_pruned].sum()
                    / (filter_scores.sum() + 1e-12)
                ).item()

                begin = layer["layer_idx"]

                if params[i + 1]["layer_type"] == "Flatten":
                    end = params[i + 2]["layer_idx"]
                else:
                    end = params[i + 1]["layer_idx"]

                flop = compute_total_flops(model.model[begin:end + 1], input_shape)
                score = err / (flop + 1e-12)

                infos.append(
                    {
                        "layer_type": layer["layer_type"],
                        "global_idx": layer["layer_idx"],
                        "idx_in_params": i,
                        "score": score,
                        "keep_rank": keep_rank,
                    }
                )

                input_shape = forward_matrix.shape[1:]

            if len(infos) == 0:
                print("No more layers can be pruned.")
                break

            best = min(infos, key=lambda x: x["score"])

            l = best["idx_in_params"]
            global_idx = best["global_idx"]
            keep_rank = best["keep_rank"]
            layer_type = best["layer_type"]

            print(
                f"-------Begin filter pruning-------\n"
                f"layer_idx: {global_idx}, "
                f"layer_type: {layer_type}, "
                f"keep_rank: {keep_rank}, "
                f"score: {best['score']:.6e}"
            )

            params = filter_pruning_l1(params, l, keep_rank)

            model = load_pruned_model(model, params)
            model.to(device)
            model.eval()

            if use_bn_recalibration and train_loader is not None:
                bn_recalibration(
                    model,
                    train_loader,
                    device,
                    num_batches=bn_batches,
                )

            F = compute_total_flops(model.model, input_shape_origin)

            params = extract_params(model.model)

        acc, _, test_loss = evaluate_pruned_model(model, test_loader, device)
        accs.append(acc)
        test_losses.append(test_loss)
        layerwise_history[f"{j:.2f}"] = get_layerwise_retention(params, original_widths)

    print(f"Flops after pruning: {F0} -> {F}, ratio = {F / F0:.4f}")

    return model, accs, test_losses, layerwise_history