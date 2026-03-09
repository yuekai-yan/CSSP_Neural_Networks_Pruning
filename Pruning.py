import torch
import torch.nn as nn
import numpy as np
import copy
from CSSP.ARP import ARP
from CSSP.StrongRRQR import sRRQR_rank
from CSSP.RPCholesky import RPCholesky

def CSSP(method, M, k):
    """
    Column Subset Selection Problem (CSSP) for matrix M.

    Inputs:
        method : str
            Method to select columns:
                "ARP" : use Adaptive Randomized Pivoting (ARP).
                "StrongRRQR" : use strong RRQR-based ID.

        M : Torch.Tensor of shape (n, m)
            Input matrix.

        k : int
            Number of columns to select.

    Outputs:
        p_tensor : Torch.Tensor of shape (k,)
            Indices of selected columns.

        T_tensor : Torch.Tensor of shape (k, m)
            Interpolation matrix obtained from RRQR-based ID.
    """
    M_np = M.detach().cpu().numpy()
    match method:
        case "StrongRRQR":
            _, _, p, _, _ = sRRQR_rank(M_np, f=2.0, k=k)
            p = p[:k]
        case "ARP":
            p = ARP(M_np, k)
        case "RPCholesky":
            p = RPCholesky(M_np, k)

    # Construct interpolation matrix T
    M_prune = M_np[:, p]   # (n, k)
    T = np.linalg.pinv(M_prune) @ M_np   # (k, m)
    p_tensor = torch.as_tensor(p, device=M.device)
    T_tensor = torch.from_numpy(T).to(device=M.device)

    return p_tensor, T_tensor


def extract_params(model):
    """
    Extract parameters from the model for the form of nn.Sequential.
    Inputs:
    model : torch.nn.Module

    Outputs:     params : list of dict, 
                        'layer_type' : str, type of the layer (e.g., 'Conv2d', 'Linear', 'Flatten')
                        'weight' : torch.Tensor, weight matrix
                        'bias' : torch.Tensor or None, bias vector       
    """
    params = []
    for idx, layer in enumerate(model):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer_info = {
                'layer_type': type(layer).__name__,    # 'Conv2d' or 'Linear'
                'layer_idx': idx,
                'weight': layer.weight,
                'bias': layer.bias
            }
            params.append(layer_info)

        elif isinstance(layer, nn.Flatten):
            layer_info = {
                'layer_type': 'Flatten',
                'layer_idx': idx,
                'weight': None,
                'bias': None
            }
            params.append(layer_info)

    return params


def forward_to_layer(model, X, l):
    """
    Forward propagate input X through the model of the form nn.sequential before layer l.
    Inputs:
        model : torch.nn.Module
            Neural network model.

        X : torch.Tensor, shape (n, d)
            Input data, where d is input dimension and n is number of samples.

        l : int
            Layer index to forward to.
    Outputs:
        out : torch.Tensor, 
              (batch_size, out_neurons) -> 'linear' 
              (batch_size, out_channels, out_height, out_width) -> 'conv' / 'conv'+'pool'    
    """
    out = X
    for i in range(l):
        out = model[i](out)
    return out



def prune_model(model0, X, keep_ratio, method, S=None, device=None):
    """
    Prune the model using CSSP-based method, for layers of type 'Linear' and 'Conv'
    Inputs:
        model0:         torch.nn.Module
        X:              torch.Tensor, (batch_size, channel, height, width)
        keep_ratio:     float, ratio of neurons to keep in each layer
        method:         str, method for CSSP 
        S:              set of layer indices not to prune
    Outputs:
        params_new:      list of dict, pruned parameters for each layer
                        layer_type:     str, type of the layer (e.g., 'Conv2d', 'Linear', 'Flatten')
                        layer_idx:      int, index of the layer in the model
                        weight:         torch.Tensor, pruned weight matrices 
                                        (out_neurons, in_neurons) -> 'linear'
                                        (out_channels, in_channels, kernel_size, kernel_size) -> 'conv'
                        bias:         torch.Tensor, pruned bias vectors
    """
    if device is None:
        try:
            device = next(model0.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    params_new = []

    # layers not to prune, e.g., output layer
    S = {len(model0.model) - 1}

    # extract weights
    params = extract_params(model0.model)

    # determine initial transformation matrix T0 for Conv
    if params[0]['layer_type'] == 'Conv2d':
        T0 = torch.eye(X.shape[1]).to(device)


    for i, layer in enumerate(params):
        if i < len(params) - 1:
            Z = forward_to_layer(model0.model, X, params[i+1]['layer_idx']) # (batch_size, out_neurons) -> 'linear'
                                        # (batch_size, out_channels, out_height, out_width) -> 'conv' / 'conv'+'pool'
                                        # (batch_size, flattened_dim) -> 'flatten'

        if layer['layer_type'] != 'Flatten':
            W = layer['weight']   # (m, d) -> 'linear'
                                # (out_channels, in_channels, kernel_size, kernel_size) -> 'conv'   
            b = layer['bias']   # (m,) -> 'linear'
                                # (out_channels,) -> 'conv'
            m = W.shape[0]
            k = int(m * keep_ratio)

            match layer['layer_type']:
                case 'Linear':
                    if layer['layer_idx'] not in S:    
                        p, T = CSSP(method, Z, k)    # T -> (k, m)
                        # construct pruned parameters
                        W_hat = W.index_select(0, p) @ T0.T   # 1st dimension of 'weight' -> output
                        b_hat = b.index_select(0, p)
                        # modify T0 to tensor for next layer
                        T0 = T 
                    else:
                        W_hat = W @ T0.T
                        b_hat = b.clone()
                        T0 = torch.eye(W.shape[0])                

                case 'Conv2d':                
                    if layer['layer_idx'] not in S:
                        Z_reshaped = Z.reshape(Z.shape[1], -1)
                        M = Z_reshaped.T
                        p, T = CSSP(method, M, k)    # T -> (k, m)
                        # construct pruned parameters
                        U = W.index_select(0, p)    # 1st dimension of 'weight' -> output
                        W_hat = torch.einsum('km, omhw -> okhw', T0, U)
                        b_hat = b.index_select(0, p)
                        # modify T0 to tensor for next layer
                        T0 = T
                    else:
                        W_hat = torch.einsum('km, omhw -> okhw', T0, W)
                        b_hat = b.clone()
                        T0 = torch.eye(W.shape[0])  # same number of output channels


        else:
            if layer['layer_idx'] == 0:
                T0 = torch.eye(Z.shape[1]).to(device)
            else:
                Z0 = forward_to_layer(model0.model, X, params[i+1]['layer_idx']-1)
                size = Z0.shape[2] * Z0.shape[3]
                I = torch.eye(size).to(device)
                T0 = torch.kron(T0, I).to(device)

        layer_info_new = {
            'layer_type': layer['layer_type'],
            'layer_idx': layer['layer_idx'],
            'weight':  W_hat if layer['weight'] is not None else None,
            'bias': b_hat if layer['bias'] is not None else None
        }
        params_new.append(layer_info_new)


    return params_new


def load_pruned_model(model0, params_new, device=None):
    """
    Load pruned parameters back into the origin model
    The overall layer order is preserved; only the corresponding
    Linear / Conv2d layers are replaced.

    Inputs:
        model0:      torch.nn.Module, original model
        params_new:  list of dict, pruned parameters for each layer
                     e.g.
                     {
                         "layer_type": str, "Linear" / "Conv2d" / "Flatten",
                         "layer_idx":  int, actual layer index,
                         "weight":     torch.Tensor, pruned weight matrices
                                       (out_neurons, in_neurons) -> 'linear'
                                       (out_channels, in_channels, kernel_size, kernel_size) -> 'conv'
                         "bias":     torch.Tensor, pruned bias vectors 
                     }

    Output:
        model:       model with pruned layers loaded
    """
    if device is None:
        try:
            device = next(model0.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model = copy.deepcopy(model0)

    seq = model.model   # the model structure is assumed to be self.model = nn.Sequential(...)

    for p in params_new:
        layer_type = p["layer_type"]
        idx = p["layer_idx"]

        match layer_type:
            case "Flatten":
                continue

            case "Linear":
                old_layer = seq[idx]

                W = p["weight"].to(device)
                b = p["bias"].to(device)

                out_features, in_features = W.shape

                new_layer = nn.Linear(
                    in_features = in_features,
                    out_features = out_features,
                ).to(device)

                with torch.no_grad():
                    new_layer.weight.copy_(W)
                    new_layer.bias.copy_(b)

                seq[idx] = new_layer

            case "Conv2d":
                old_layer = seq[idx]

                W = p["weight"].to(device)
                b = p["bias"].to(device)

                out_channels, in_channels, kH, kW = W.shape

                new_layer = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kH, kW),
                    stride=old_layer.stride,
                    padding=old_layer.padding,
                    dilation=old_layer.dilation,
                    groups=old_layer.groups,
                    padding_mode=old_layer.padding_mode
                ).to(device)

                with torch.no_grad():
                    new_layer.weight.copy_(W)
                    new_layer.bias.copy_(b)

                seq[idx] = new_layer


    return model



def evaluate_pruned_model(pruned_model, test_data, device=None):
    """
    Evaluate the pruned model on test_data

    Inputs:
        pruned_model:  pruned PyTorch model
        test_data:     DataLoader for test set
        device:        target device; if None, use the model device

    Output:
        accuracy:      classification accuracy on test_data
        wrong_samples: List of tuples (index, predicted_label) for misclassified samples
    """
    if device is None:
        try:
            device = next(pruned_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    pruned_model = pruned_model.to(device)
    pruned_model.eval()

    correct = 0
    total = 0

    wrong_samples = []
    base_idx = 0

    with torch.no_grad():
        for X, y in test_data:
            X = X.to(device)
            y = y.to(device)

            logits = pruned_model(X)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            # Record indices of misclassified samples
            wrong_in_batch = torch.where(pred != y)[0]
            wrong_preds = pred[wrong_in_batch]
            wrong_indices = base_idx + wrong_in_batch
            for idx, p in zip(wrong_indices, wrong_preds):
                wrong_samples.append((idx.item(), p.item()))
           
            base_idx += y.size(0)

    accuracy = correct / total
    return accuracy, wrong_samples