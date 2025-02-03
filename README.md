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
A BSDE is defined on a filtered probability space $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t \in [0, T]}, P) \colon$


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

The DeepBSDE framework employs a systematic process to solve backward stochastic differential equations (BSDEs), divided into three key steps:

1. **Time Discretization**  
   The continuous time interval \[0, T] is divided into a finite number of discrete time steps. This discretization ensures computational feasibility while approximating the dynamics of the stochastic process.

2. **Neural Network Approximation**  
   Neural networks are used to approximate the conditional expectations inherent in the BSDE formulation. Specifically, the \(Z_t\) term is parameterized as a function of the state variable at each discrete time step, allowing flexible and accurate representation.

3. **Model Training**  
   The model is trained by minimizing a loss function. This loss measures the difference between the predicted terminal condition and the true terminal condition. The optimization process averages the loss over multiple simulated trajectories, using stochastic gradient-based techniques to adjust the neural network parameters effectively.



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
