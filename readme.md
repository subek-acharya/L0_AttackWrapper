# L0 Attack Wrapper

PyTorch implementation of L0-norm constrained adversarial attacks for evaluating neural network robustness.

## Overview

This repository contains implementations of three sparse adversarial attack methods based on the paper:

**"Sparse and Imperceivable Adversarial Attacks"**  
*Francesco Croce, Matthias Hein*  
*University of Tübingen*  
*ICCV 2019*

**Original Paper Implementation:** [https://github.com/fra31/sparse-imperceivable-attacks](https://github.com/fra31/sparse-imperceivable-attacks)

## Attack Methods

### 1. L0 PGD Attack (`L0_PGD_AttackWrapper.py`)

Standard L0-norm constrained Projected Gradient Descent attack that limits the number of perturbed pixels (sparsity constraint).

**Key Features:**
- **Sparsity Control**: Only modifies a fixed number `k` of pixels
- **Random Initialization**: Starts with random perturbations
- **Iterative Optimization**: Uses PGD to find adversarial examples
- **Projection**: Projects perturbations to satisfy pixel values in range [0, 1] and maximum k pixels modified per image

### 2. L0-L∞ PGD Attack (`L0_Linf_PGD_AttackWrapper.py`)

Combines L0 sparsity constraint with L∞ magnitude constraint, ensuring perturbations are both sparse and small in magnitude.

**Key Features:**
- **Dual Constraint**: 
  - L0: Maximum `k` pixels modified
  - L∞: Each pixel perturbed by at most `ε` (epsilon)
- **Epsilon Constraint**: `||perturbation||_∞ ≤ ε`
- **Tighter Bounds**: More restrictive than standard L0

### 3. L0-Sigma PGD Attack (`L0_Sigma_PGD_AttackWrapper.py`)

**Most sophisticated attack** that adapts perturbation bounds based on local image structure using a sigma-map.

**Key Features:**
- **Adaptive Perturbations**: Perturbation magnitude varies per pixel based on local variance
- **Sigma-Map**: Computes local standard deviation for each pixel from neighboring pixels
- **Multiplicative Perturbations**: `x_adv = x_nat * (1 + λ * σ)` where `λ ∈ [-κ, κ]`
- **Structure-Aware**: Larger perturbations in high-variance regions, smaller in smooth regions

## Project Structure

```
L0_AttackWrapper/
├── L0_PGD_AttackWrapper.py          # Standard L0 attack
├── L0_Linf_PGD_AttackWrapper.py     # L0 + L∞ attack
├── L0_Sigma_PGD_AttackWrapper.py    # L0 + Sigma attack (structure-aware)
├── DataManagerPytorch.py             # Data loading utilities
├── L0_Utils.py                       # Helper functions
├── Utils.py                          # General utilities
├── resnet.py                         # ResNet model architecture
├── main.py                           # Main execution script
├── models/                           # Trained model checkpoints
│   └── model_test.pt
└── data/      
```                       # Dataset directory

## Installation Requirements

pip install torch torchvision numpy