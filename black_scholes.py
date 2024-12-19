# Black-Scholes model parameters and functions
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
