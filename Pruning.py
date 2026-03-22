import torch
import torch.nn as nn
import numpy as np
import copy
from CSSP.ARP import ARP
from CSSP.StrongRRQR import sRRQR_rank, sRRQR_tol
from CSSP.RPCholesky import RPCholesky, RPCholesky_tol
from scipy.linalg import qr

def CSSP(method, M, criterion, value):
    """
    Column Subset Selection Problem (CSSP) for matrix M.

    Inputs:
        method:         str, CSSP method
        M:              Torch.Tensor of shape (n, m)
        criterion:      str, "keep_rank" or "tol"
        value:          expected rank k or tol

    Outputs:
        p_tensor : Torch.Tensor of shape (k,)
            Indices of selected columns.

        T_tensor : Torch.Tensor of shape (k, m)
            Interpolation matrix obtained from RRQR-based ID.
    """
    M_np = M.detach().cpu().numpy()
    if criterion == 'tol':
        match method:
            case "StrongRRQR":
                _, _, p, k, _ = sRRQR_tol(M_np, f=2.0, tol=value)
                p = p[:k]
            case "ARP":
                p, k = ARP(M_np, tol=value)
            case "RPCholesky":
                p, k = RPCholesky_tol(M_np, tol=value)
    else:
        match method:
            case "StrongRRQR":
                _, _, p, k, _ = sRRQR_rank(M_np, f=2.0, k=value)
                p = p[:k]
            case "ARP":
                p, k = ARP(M_np, value)
            case "RPCholesky":
                p, k = RPCholesky(M_np, value)

    # Construct interpolation matrix T
    M_prune = M_np[:, p]   # (n, k)
    T = np.linalg.pinv(M_prune) @ M_np   # (k, m)
    p_tensor = torch.as_tensor(p, device=M.device)
    T_tensor = torch.from_numpy(T).to(device=M.device)

    return p_tensor, T_tensor, k


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

def forward_from_a_to_b(model, X, a, b):
    """
    Forward propagate X through model layers from index a to b (exclusive).

    Inputs:
        model : torch.nn.Module or nn.Sequential
            Neural network model.
        X : torch.Tensor
            Input tensor.
        a : int
            Start layer index.
        b : int
            End layer index (inclusive).

    Outputs:
        out : torch.Tensor
            Output after passing through layers a to b.
    """
    out = X
    for i in range(a, b):
        out = model[i](out)
    return out


def prune_model(model0, X, prune_info, S=None, device=None):
    """
    Prune the model using CSSP-based method, for layers of type 'Linear' and 'Conv'
    Inputs:
        model0:         torch.nn.Module
        X:              torch.Tensor, (batch_size, channel, height, width)
        method:         str, method for CSSP
        prune_info:     dict of dict, pruning information for each layer
                        e.g. prune_info = {
                                            3: {"criterion": "keep_rank", "value": 0.5, "method": "StrongRRQR"},
                                            5: {"criterion": "tol", "value": 1e-4, "method": "RPCholesky"},
                                        }
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
            if layer['layer_idx'] not in S:
                print(f"-------Begin pruning-------\nlayer_idx: {layer['layer_idx']}, layer_type: {layer['layer_type']}")
            W = layer['weight']   # (m, d) -> 'linear'
                                # (out_channels, in_channels, kernel_size, kernel_size) -> 'conv'   
            b = layer['bias']   # (m,) -> 'linear'
                                # (out_channels,) -> 'conv'
            m = W.shape[0]

            cfg = prune_info.get(layer['layer_idx'])

            if cfg is not None:
                criterion = cfg["criterion"]
                value = cfg["value"]
                method = cfg["method"]

                if criterion == "keep_rank":
                    value = int(m * value)

            match layer['layer_type']:
                case 'Linear':
                    if layer['layer_idx'] not in S:
                        p, T, k = CSSP(method, Z, criterion, value)    # T -> (k, m)
                        # construct pruned parameters
                        W_hat = W.index_select(0, p) @ T0.T   # 1st dimension of 'weight' -> output
                        b_hat = b.index_select(0, p)
                        # modify T0 to tensor for next layer
                        T0 = T
                        print(f"number of columns: {m} -> {k}") 
                    else:
                        W_hat = W @ T0.T
                        b_hat = b.clone()
                        T0 = torch.eye(W.shape[0])
                    
                                    
                case 'Conv2d':                
                    if layer['layer_idx'] not in S:
                        Z_reshaped = Z.reshape(Z.shape[1], -1)
                        M = Z_reshaped.T
                        p, T, k = CSSP(method, M, criterion, value)    # T -> (k, m)
                        # construct pruned parameters
                        U = W.index_select(0, p)    # 1st dimension of 'weight' -> output
                        W_hat = torch.einsum('km, omhw -> okhw', T0, U)
                        b_hat = b.index_select(0, p)
                        # modify T0 to tensor for next layer
                        T0 = T
                        print(f"number of out_channels: {m} -> {k}")
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



def compute_total_flops(model: nn.Sequential, input_shape):
    """
    Compute the total FLOPs for an arbitrary nn.Sequential model.

    Supported layers:
        - nn.Conv2d    ->    2 * in_channels * out_channels * k_h * k_w * out_h * out_w
        - nn.Linear    ->    2 * layer.in_features * layer.out_features
        - nn.ReLU      ->    out.numel()
        - nn.MaxPool2d ->    out.numel() * (k_h * k_w)
        - nn.Flatten   ->    0

    Inputs:
        model: an nn.Sequential model
        input_shape: tuple, e.g. (1, 28, 28)

    Outputs:
        total_flops: total FLOPs of the model
    """
    total_flops = 0
    x = torch.zeros(1, *input_shape, device=next(model.parameters()).device)    # (batch_size, channels, height, width)

    for layer in model:
        if isinstance(layer, nn.Conv2d):
            out = layer(x)

            _, out_channels, out_h, out_w = out.shape
            k_h, k_w = layer.kernel_size
            in_channels = layer.in_channels

            layer_flops = 2 * (
                in_channels * out_channels * k_h * k_w * out_h * out_w
            )

            total_flops += layer_flops
            x = out

        elif isinstance(layer, nn.Linear):
            if x.dim() > 2:
                x = x.view(x.size(0), -1)

            out = layer(x)
            layer_flops = 2 * layer.in_features * layer.out_features

            total_flops += layer_flops
            x = out

        elif isinstance(layer, nn.ReLU):
            out = layer(x)
            layer_flops = out.numel()

            total_flops += layer_flops
            x = out

        elif isinstance(layer, nn.MaxPool2d):
            out = layer(x)

            if isinstance(layer.kernel_size, tuple):
                k_h, k_w = layer.kernel_size
            else:
                k_h = k_w = layer.kernel_size

            layer_flops = out.numel() * (k_h * k_w)

            total_flops += layer_flops
            x = out

        elif isinstance(layer, nn.Flatten):
            x = layer(x)

        else:
            raise TypeError(f"Unsupported layer type: {layer.__class__.__name__}")

    return total_flops




def iterative_pruning(model0, X, input_shape, rho, step_size, method, S=None, device=None):
    """
    Prune the model using CSSP-based method, for layers of type 'Linear' and 'Conv'
    Inputs:
        model0:         torch.nn.Module
        X:              torch.Tensor, (batch_size, channel, height, width)
        method:         str, method for CSSP
        input_shape:    tuple, shape of the input image, e.g. (1, 28, 28), (3, 32, 32)
        rho:            float, flop ratio
        step_size:      float, keep_ratio for each layer
        method:         str, CSSP method
        S:              set of layer indices not to prune

    Outputs:
        model:          torch.nn.Module, model after iterative pruning
    """
    if device is None:
        try:
            device = next(model0.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # layers not to prune, e.g., output layer
    S = {len(model0.model) - 1}

    # extract weights
    params = extract_params(model0.model)
    # store the input_shape
    input_shape_origin = input_shape

    F = compute_total_flops(model0.model, input_shape_origin)
    F0 = F

    # initialize model
    model = model0

    while F > F0 * rho:
        infos = []    # list of dict to store the prunability infos for each layer
        input_shape = input_shape_origin
        forward_matrix = X
        # compute the score for every layer needed to be pruned
        for i, layer in enumerate(params[:-1]):

            if layer['layer_idx'] in S or layer['layer_type'] == 'Flatten':
                forward_matrix = forward_from_a_to_b(model.model, forward_matrix, layer['layer_idx'], params[i+1]['layer_idx'])
                input_shape = forward_matrix.shape[1:]
                continue
            
            forward_matrix = forward_from_a_to_b(model.model, forward_matrix, layer['layer_idx'], params[i+1]['layer_idx']) 
                                        # (batch_size, out_neurons) -> 'linear'
                                        # (batch_size, out_channels, out_height, out_width) -> 'conv' / 'conv'+'pool'

            W = layer['weight']   # (m, d) -> 'linear'
                                # (out_channels, in_channels, kernel_size, kernel_size) -> 'conv'
        
            if layer['layer_type'] == 'Linear':
                M = forward_matrix.detach().cpu().numpy()
                _, R, _ = qr(M, mode="economic", pivoting=True)    # Z[:, P] = Q @ R
            elif layer['layer_type'] == 'Conv2d':
                forward_reshaped = forward_matrix.reshape(forward_matrix.shape[1], -1)
                M = forward_reshaped.detach().cpu().numpy()
                _, R, _ = qr(M.T, mode="economic", pivoting=True)

            k = int(W.shape[0] * step_size)
            err = abs(R[k, k] / R[0, 0])

            # compute flop
            begin = layer['layer_idx']

            if params[i+1]['layer_idx'] == 'Flatten':
                end = params[i+2]['layer_idx']
            else:
                end = params[i+1]['layer_idx']
            
            flop = compute_total_flops(model.model[begin: end+1], input_shape)

            input_shape = forward_matrix.shape[1:]

            info = {
                'layer_type': layer['layer_type'],
                'global_idx': layer['layer_idx'],
                'idx_in_params': i,
                'score': err / flop,
                'keep_rank': k
            }
            infos.append(info)

        best = min(infos, key=lambda x: x['score'])
        l = best['idx_in_params']
        global_idx = best['global_idx']
        keep_rank = best['keep_rank']
        layer_type = best['layer_type']

        print(f"-------Begin pruning-------\nlayer_idx: {global_idx}, layer_type: {layer_type}")


        W = params[l]['weight']   # (m, d) -> 'linear'
                            # (out_channels, in_channels, kernel_size, kernel_size) -> 'conv'   
        b = params[l]['bias']   # (m,) -> 'linear'
                            # (out_channels,) -> 'conv'
        m = W.shape[0]

        Z = forward_to_layer(model.model, X, params[l+1]['layer_idx']) # (batch_size, out_neurons) -> 'linear'
                            # (batch_size, out_channels, out_height, out_width) -> 'conv' / 'conv'+'pool'

        match layer_type:
            case 'Linear':
                p, T, k = CSSP(method, Z, 'keep_rank', keep_rank)    # T -> (k, m)
                # construct pruned parameters
                W_hat = W.index_select(0, p)   # 1st dimension of 'weight' -> output
                b_hat = b.index_select(0, p)
                print(f"number of out_neurons: {m} -> {k}")
                print() 
                
                                
            case 'Conv2d':                
                Z_reshaped = Z.reshape(Z.shape[1], -1)
                M = Z_reshaped.T
                p, T, k = CSSP(method, M, 'keep_rank', keep_rank)    # T -> (k, m)
                # construct pruned parameters
                W_hat = W.index_select(0, p)    # 1st dimension of 'weight' -> output
                b_hat = b.index_select(0, p)
                print(f"number of out_channels: {m} -> {k}")
                print()


        current_layer_info_new = {
            'layer_type': layer_type,
            'layer_idx': global_idx,
            'weight':  W_hat if W is not None else None,
            'bias': b_hat if b is not None else None
        }
        params[l] = current_layer_info_new

        # deal with the effect on the next layer
        layer_type_next = params[l+1]['layer_type']
        global_idx_next = params[l+1]['layer_idx']

        if layer_type_next != 'Flatten':

            W_next = params[l+1]['weight']   # (out_neurons, in_neurons) -> 'linear'
                            # (out_channels, in_channels, kernel_size, kernel_size) -> 'conv'   
            b_next = params[l+1]['bias']   # (out_neurons,) -> 'linear'
                            # (out_channels,) -> 'conv'

            match layer_type_next:
                case 'Linear':
                    W_hat_next = W_next @ T.T    # in_neurons: m -> k, i.e. (out_neurons, m) -> (out_neurons, k)
                    b_hat_next = b_next.clone()                    
                                    
                case 'Conv2d':                
                    W_hat_next = torch.einsum('km, omhw -> okhw', T, W_next)
                    b_hat_next = b_next.clone()
            
            next_layer_info_new = {
                'layer_type': layer_type_next,
                'layer_idx': global_idx_next,
                'weight':  W_hat_next if W_next is not None else None,
                'bias': b_hat_next if b_next is not None else None
            }
            params[l+1] = next_layer_info_new


        else:
            size = Z.shape[2] * Z.shape[3]
            I = torch.eye(size).to(device)
            T = torch.kron(T, I).to(device)    # (k * size, m * size)
            # next layer type must be 'Linear'
            layer_type_next = params[l+2]['layer_type']
            global_idx_next = params[l+2]['layer_idx']
            W_next = params[l+2]['weight']    # (out_neurons, m * size)
            b_next = params[l+2]['bias']      # (out_neurons,)

            W_hat_next = W_next @ T.T         # (out_neurons, k * size)
            b_hat_next = b_next.clone()

            next_layer_info_new = {
                'layer_type': layer_type_next,
                'layer_idx': global_idx_next,
                'weight':  W_hat_next if W_next is not None else None,
                'bias': b_hat_next if b_next is not None else None
            }
            params[l+2] = next_layer_info_new


        model = load_pruned_model(model, params)
        F = compute_total_flops(model.model, input_shape_origin)

    print(f"Flops after pruning: {F0} -> {F}")

    return model











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