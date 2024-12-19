# DeepBSDE

A Neural Network-Based Model for Backward Stochastic Differential Equations

## Overview
Backward Stochastic Differential Equations (BSDEs) play a pivotal role in quantitative finance, stochastic control, and mathematical optimization. The DeepBSDE framework leverages neural networks to approximate solutions to BSDEs and Forward-Backward Stochastic Differential Equations (FBSDEs) efficiently, providing a powerful tool to address problems that are intractable using traditional numerical methods.

### Key Features
- Neural network-based approach for solving BSDEs and FBSDEs.
- Compatible with **tensorflow==2.x**, **numpy==1.24**, **matplotlib==3.6**, and **scipy==1.10**.
- Designed to address numerical challenges inherent in backward integration and conditional expectation estimation.
- Applications include European option pricing, stochastic control, and optimization problems.

## Theoretical Background

### BSDE Definition
A BSDE is defined on a filtered probability space `((\Omega, (\mathcal{F}_t)_{t \in [0, T]}, P))`:

$$
Y_t = \xi + \int_t^T f(s, Y_s, Z_s) \, ds - \int_t^T Z_s \, dW_s, \quad t \in [0, T].
$$

Here, `Y_t` and `Z_t` are the processes to solve, `W_t` is a `d`-dimensional Brownian motion, and `f` is a driver function.

### Forward-Backward Systems
FBSDEs extend BSDEs by coupling them with a forward SDE:

$$
X_t = x + \int_0^t \mu(s, X_s) \, ds + \int_0^t \sigma(s, X_s) \, dW_s, \quad Y_t = \Phi(X_T) + \int_t^T f(s, X_s, Y_s, Z_s) \, ds - \int_t^T Z_s \, dW_s.
$$

### Theoretical Insights
- **Existence and Uniqueness:** Solutions exist under Lipschitz continuity and boundedness of coefficients.
- **Equivalence to PDEs:** BSDEs are equivalent to certain parabolic PDEs, enabling classical interpretation:

$$
\left(\frac{\partial}{\partial t} + \mathcal{L}_t\right)u(t,x) + f\big(t, x, u(t,x), \nabla u(t,x)^\top \sigma(t,x)\big) = 0, \quad u(T,x) = \Phi(x).
$$

## Numerical Methods

### Challenges
Solving BSDEs numerically is challenging due to:
1. Backward integration.
2. Coupled conditional expectations.

### Euler--Maruyama for Forward SDEs
$$
X_{t_{i+1}} = X_{t_i} + \mu(t_i, X_{t_i}) \Delta t + \sigma(t_i, X_{t_i}) \Delta W_i.
$$

### DeepBSDE Approach

1. Discretize time \( 0 = t_0 < t_1 < \cdots < t_n = T \).
2. Use neural networks \( Z_{t_k}^{\theta_k} \) to approximate \( Z_{t_k} \):
   \[
   \bar{Z}_{t_k} = Z_{t_k}^{\theta_k}(\bar{X}_{t_k}).
   \]
3. Train networks by minimizing the loss:
   \[
   \inf_{\Theta} \frac{1}{M} \sum_{l=1}^M \mathbb{E}\big[|\Phi(\bar{X}_{T,l}) - \bar{Y}_{T,l}^\Theta|^2\big].
   \]


## Applications

### European Option Pricing
- **Model:** Risk-free interest rate `r_t`, risky asset `dS_t = S_t, mu_t, dt + S_t, sigma_t, dW_t`, payoff `xi`.
- **BSDE Formulation:**

$$
Y_t = \xi - \int_t^T \big(Z_s \pi_s + r_s\big) \, ds - \int_t^T Z_s \, dW_s.
$$

### Stochastic Control
- **Optimization Problem:**

$$
\sup_k J(k), \quad J(k) = \mathbb{E}\left[\Phi(X_T) + \int_0^T f(s, X_s, k_s) \, ds\right].
$$

- **FBSDE Relation:**

$$
Y_t = \Phi(X_T) + \int_t^T f(s, X_s, k_s) \, ds - \int_t^T Z_s \, dB_s.
$$


## Installation

```bash
pip install tensorflow==2.x numpy==1.24 matplotlib==3.6 scipy==1.10

## Installation

```bash
pip install tensorflow==2.x numpy==1.24 matplotlib==3.6 scipy==1.10
```

## Usage

### Training the Model
```python
import tensorflow as tf
import numpy as np
from deepbsde import DeepBSDE

# Define problem parameters
T = 1.0  # Time horizon
n = 50   # Time steps

# Initialize model
model = DeepBSDE(T, n)

# Train
model.train(epochs=100, batch_size=64)
```

### Visualizing Results
```python
import matplotlib.pyplot as plt

# Plot results
plt.plot(model.time_grid, model.solution, label="DeepBSDE Solution")
plt.legend()
plt.show()
```
