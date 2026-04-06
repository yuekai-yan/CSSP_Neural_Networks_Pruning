# Column subset selection methods in NLA with application to neural networks pruning

## Overview

Our pruning approach is built upon **column subset selection (CSS)**, a classical technique from **numerical linear algebra**. By selecting informative columns from weight matrices, the method preserves their essential structure and achieves **structured pruning** through the reduction of layer or channel dimensions. This structured design makes the approach naturally compatible with other model compression techniques, including **knowledge distillation** and **quantization**.

Therefore, it is particularly well suited to real-world deployment scenarios involving **edge devices**, **mobile platforms**, **embedded AI systems**, and other resource-constrained environments where inference efficiency and hardware-aware model design are of central importance. This project is motivated by the framework proposed by Chee et al. in [Model Preserving Compression for Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2022/file/f8928b073ccbec15d35f2a9d39430bfd-Paper-Conference.pdf), and further explores alternative pruning strategies in this context.

## Method

We use an iterative pruning strategy, which means that there are 2 steps in each pruning loop: the first is selecting the most prunable layer / channel by computing a `prunability score`, and the second is pruning this selected layer / channel using different CSS methods (e.g. StrongRRQR, ARP, RPCholesky).

The essence of computing the `prunability score` lies in first computing an inexpensive QR decomposition of the corresponding forward matrix for each layer, which means that we can design different ways of such computation for different CSS methods in order to better align with the pruning process in the second step.

Specifically, we use:

- **StrongRRQR** → **Pivoted QR**
- **ARP** → **Sketched Pivoted QR**
- **RPCholesky** → **Pivoted QR**