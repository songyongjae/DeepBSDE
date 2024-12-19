from model import DeepBSDE
from utils import BS_CALL
import tensorflow as tf

# Define problem-specific parameters
m = 0.08
r = 0.03
s = 0.2
x = 100.

def f(t, x, y, z):
    return -z * (m - r) / s - r * y

def mu(t, x):
    return x * m

def sigma(t, x):
    return x * s

def g(x):
    return tf.maximum(x - 110, 0)

# Initialize the model
model = DeepBSDE(f=f, mu=mu, sigma=sigma, g=g, x=x, dim=1)

# Training function
def train_model(model, num_iterations, batch_size):
    for i in range(num_iterations):
        inp = model.draw_X_and_dW(batch_size)
        loss = model.train(inp)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy()}")
    return model

# Train the model
trained_model = train_model(model, num_iterations=3000, batch_size=30)