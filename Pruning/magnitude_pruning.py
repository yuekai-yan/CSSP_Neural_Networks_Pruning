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

def get_prunable_weight(module):
    """
    Return weight tensor for:
    - nn.Conv2d
    - nn.Linear
    - ConvBNReLU
    - LinearBNReLU
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return module.weight

    if hasattr(module, "block"):
        if isinstance(module.block[0], (nn.Conv2d, nn.Linear)):
            return module.block[0].weight

    raise AttributeError(f"{type(module).__name__} has no prunable weight")


def iterative_magnitude_pruning(
    model0,
    input_shape,
    rho,
    step_size,
    test_loader,
    train_loader=None,
    S=None,
    device=None,
    use_bn_recalibration=False,
    bn_batches=20,
):
    """
    Iterative unstructured magnitude pruning.
    Supports both wrapped layers and raw Conv2d / Linear layers.

    Stop criterion: nonzero parameter ratio only.
    """

    if device is None:
        try:
            device = next(model0.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model = copy.deepcopy(model0).to(device)
    model.eval()

    if S is None:
        S = {len(model.model) - 1}

    params = extract_params(model.model)

    masks = {}

    for layer in params:
        if layer["layer_type"] in ["Conv2d", "ConvBNReLU", "Linear", "LinearBNReLU"]:
            idx = layer["layer_idx"]

            if idx in S:
                continue

            module = model.model[idx]
            weight = get_prunable_weight(module).data

            masks[idx] = torch.ones_like(weight, device=device)

    def count_nonzero_params():
        total = 0
        for p in model.parameters():
            total += torch.count_nonzero(p).item()
        return total

    P0 = sum(p.numel() for p in model.parameters())
    P = count_nonzero_params()

    accs = []
    test_losses = []

    for target_ratio in rho:
        while P > P0 * target_ratio:
            infos = []
            params = extract_params(model.model)

            for layer in params:
                layer_type = layer["layer_type"]

                if layer_type not in ["Conv2d", "ConvBNReLU", "Linear", "LinearBNReLU"]:
                    continue

                global_idx = layer["layer_idx"]

                if global_idx in S:
                    continue

                if global_idx not in masks:
                    continue

                module = model.model[global_idx]
                W = get_prunable_weight(module).data
                mask = masks[global_idx]

                alive = mask.bool()
                num_alive = alive.sum().item()

                if num_alive <= 1:
                    continue

                keep_num = int(num_alive * step_size)

                if keep_num < 1 or keep_num >= num_alive:
                    continue

                alive_weights = W[alive].abs()
                num_pruned = num_alive - keep_num

                sorted_scores, _ = torch.sort(alive_weights)

                err = (
                    sorted_scores[:num_pruned].sum()
                    / (alive_weights.sum() + 1e-12)
                ).item()

                resource = num_pruned
                score = err / (resource + 1e-12)

                infos.append(
                    {
                        "global_idx": global_idx,
                        "layer_type": layer_type,
                        "score": score,
                        "keep_num": keep_num,
                        "num_pruned": num_pruned,
                    }
                )

            if len(infos) == 0:
                print("No more weights can be pruned.")
                break

            best = min(infos, key=lambda x: x["score"])

            global_idx = best["global_idx"]
            keep_num = best["keep_num"]

            print(
                f"-------Begin magnitude pruning-------\n"
                f"layer_idx: {global_idx}, "
                f"layer_type: {best['layer_type']}, "
                f"keep_num: {keep_num}, "
                f"num_pruned: {best['num_pruned']}, "
                f"score: {best['score']:.6e}"
            )

            module = model.model[global_idx]
            W = get_prunable_weight(module).data
            mask = masks[global_idx]

            alive = mask.bool()
            alive_indices = alive.nonzero(as_tuple=False)
            alive_values = W[alive].abs()

            topk_idx = torch.topk(
                alive_values,
                keep_num,
                largest=True,
            ).indices

            new_mask = torch.zeros_like(mask)
            selected_indices = alive_indices[topk_idx]
            new_mask[tuple(selected_indices.t())] = 1.0

            masks[global_idx] = new_mask

            W.mul_(new_mask)

            model.eval()

            if use_bn_recalibration and train_loader is not None:
                bn_recalibration(
                    model,
                    train_loader,
                    device,
                    num_batches=bn_batches,
                )

            P = count_nonzero_params()

        acc, _, test_loss = evaluate_pruned_model(model, test_loader, device)

        accs.append(acc)
        test_losses.append(test_loss)

        print(
            f"Target params ratio: {target_ratio:.2f}, "
            f"Current nonzero params ratio: {P / P0:.4f}, "
            f"Accuracy: {acc:.4f}"
        )

    print(f"Nonzero params after pruning: {P0} -> {P}, ratio = {P / P0:.4f}")

    return model, accs, test_losses