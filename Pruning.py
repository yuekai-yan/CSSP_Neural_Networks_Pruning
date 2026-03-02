import torch
import numpy as np
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

        M : ndarray of shape (n, m)
            Input matrix.

        k : int
            Number of columns to select.

    Outputs:
        p : ndarray of shape (k,)
            Indices of selected columns.

        T : ndarray of shape (k, m)
            Interpolation matrix obtained from RRQR-based ID.
    """
    match method:
        case "StrongRRQR":
            _, _, p, _, _ = sRRQR_rank(M, f=2.0, k=k)
            p = p[:k]
        case "ARP":
            p = ARP(M, k)
        case "RPCholesky":
            p = RPCholesky(M, k)

    # Construct interpolation matrix T
    M_prune = M[:, p]   # (n, k)
    T = np.linalg.pinv(M_prune) @ M   # (k, m)

    return p, T

def prune_model(model0, X, keep_ratio, method):
    """
    For a model of the form:
        Flatten -> Dense(m, relu) -> Dense(10, softmax)
    
    Inputs:
        model0 : torch.nn.Module
            Trained neural network model.

        X : torch.Tensor, shape (d, n)
            Unlabeled pruning dataset, where d is input dimension
            and n is number of samples.

        keep_ratio : float, optional (default=0.7)
            Fraction of hidden neurons to keep.

        f : float, optional (default=2.0)
            Parameter controlling the strong RRQR bound.
        
        method :
            Method to select neurons:
                "ARP" : use Adaptive Randomized Pivoting (ARP) for CSSP.
                "StrongRRQR" : use strong RRQR-based ID for CSSP.

    Outputs:
        coefficients : list of torch.Tensor
            List of pruned weight matrices:
                coefficients[0] : W1_hat, shape (k, d)
                coefficients[1] : W2_hat, shape (c, k)

        biases : list of torch.Tensor
            List of pruned bias vectors:
                biases[0] : b1_hat, shape (k,)
                biases[1] : b2_hat, shape (c,)

        p : ndarray of shape (k,)
            Indices of selected hidden neurons.

        T : ndarray of shape (k, m)
            Interpolation matrix obtained from RRQR-based ID.
    """
    n = X.shape[1]
    coefficients = []
    biases = []
    # ------- Extract weights -------
    first_layer = model0.model[1]
    W1 = first_layer.weight
    b1 = first_layer.bias

    second_layer = model0.model[3]
    W2 = second_layer.weight
    b2 = second_layer.bias

    m, d = W1.shape
    c = W2.shape[0]

    # ------- Determine number of neurons to keep -------
    k = int(m * keep_ratio)

    # ------- Add bias to W and X -------
    W1_aug = torch.hstack([W1, b1.unsqueeze(1)])       # (m, d+1)
    X_aug = torch.vstack([X, torch.ones(1, X.shape[1], device=X.device)])    # (d+1, n)

    # ------- Z = relu(W^T X) -------
    Z = torch.relu(W1_aug @ X_aug)   # (m, n)

    # ------- Perform ID on Z^T -------
    M = Z.T    # (n, m)
    M_np = M.detach().numpy()
    p, T = CSSP(method, M_np, k)

    # ------- Construct pruned parameters -------
    # First layer: select subset of neurons
    p_tensor = torch.as_tensor(p, dtype=torch.long, device=W1.device)
    W1_hat = W1.index_select(0, p_tensor)      # (k, d)
    coefficients.append(W1_hat)
    b1_hat = b1.index_select(0, p_tensor)         # (k,)
    biases.append(b1_hat)

    # Second layer update
    T_tensor = torch.from_numpy(T).to(device=W2.device, dtype=W2.dtype)
    W2_hat = W2 @ T_tensor.T        # (c, k)
    coefficients.append(W2_hat)
    b2_hat = b2.clone()     # (c,)
    biases.append(b2_hat)

    return coefficients, biases, p, T


def evaluate_pruned(coeffs, biases, dataloader, device):
    W1_hat, W2_hat = coeffs
    b1_hat, b2_hat = biases

    criterion = torch.nn.CrossEntropyLoss()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 如果原模型有 Flatten
            imgs = imgs.view(imgs.size(0), -1)   # (batch_size, d)

            h = torch.relu(imgs @ W1_hat.T + b1_hat)
            outputs = h @ W2_hat.T + b2_hat

            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples



def build_pruning_matrix(loader, device=None):

    X_list = []

    for imgs, _ in loader:
        imgs = imgs.view(imgs.size(0), -1)  # flatten
        X_list.append(imgs)

    X_all = torch.cat(X_list, dim=0)  # (N, d)
    X_all = X_all.T  # (d, n)

    if device is not None:
        X_all = X_all.to(device)
    
    return X_all