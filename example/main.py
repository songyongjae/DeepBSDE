from american_option_utils import AmericanOptionUtils
from model import DeepBSDE
from training import train_model
import numpy as np
import tensorflow as tf

# Define functions for DeepBSDE

def f(t, x, y, z):
    """
    Generator function f in the BSDE.
    """
    return -0.5 * tf.reduce_sum(tf.square(z), axis=1, keepdims=True)

def g(x):
    """
    Terminal condition function.
    """
    return tf.reduce_max(x, axis=1, keepdims=True)

def sigma(t, x):
    """
    Diffusion coefficient.
    """
    return tf.linalg.diag(tf.ones_like(x))

def mu(t, x):
    """
    Drift coefficient.
    """
    return tf.zeros_like(x)

# Initialize American Option Utils
american_option = AmericanOptionUtils(initial=100, T=0.25, volatility=0.3, periods=15, r=0.02, div=0.01, strike=110)

# Calculate American option prices
call_price = american_option.american_option_price('call')
put_price = american_option.american_option_price('put')
print(f"American Call Option Price: {call_price}")
print(f"American Put Option Price: {put_price}")

# Initialize DeepBSDE model
dim = 100
x0 = np.zeros((1, dim), dtype=np.float32)
bsde_model = DeepBSDE(f=f, x=x0, g=g, sigma=sigma, mu=mu, dim=dim, time_steps=20)

# Train DeepBSDE model
print("Training DeepBSDE model...")
train_model(bsde_model, num_iterations=1000)

# Compare American option price with DeepBSDE
print("Comparing with DeepBSDE...")
bsde_price = american_option.compare_with_bsde(bsde_model, num_samples=1000)
print(f"DeepBSDE Price: {bsde_price}")