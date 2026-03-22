import matplotlib.pyplot as plt
import numpy as np

def plot_pruning_curve(base_acc, ratios, accs, labels, title=None):
    """
    Plot keep_ratio vs test accuracy curve for multiple methods.

    Inputs:
        base_acc : float
        ratios   : iterable of length m
        accs     : array-like of shape (n, m)
        labels   : iterable of length n
        title    : optional title string
    """

    n_methods = len(accs)

    plt.figure()

    for i in range(n_methods):
        plt.plot(ratios, accs[i], marker="o", label=labels[i])

    # baseline
    plt.axhline(base_acc, linestyle="--", label="Baseline")

    plt.xlabel("flops remaining")
    plt.ylabel("test_accuracy")

    if title is None:
        title = "flops remaining vs test accuracy"
    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.show()