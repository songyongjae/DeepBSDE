import tensorflow as tf

m = 0.08
r = 0.03
s = 0.2

def f(t, x, y, z):
    f_ = -z * (m - r) / s - r * y
    return f_

def mu(t, x):
    return x * m

def sigma(t, x):
    return x * s

def g(x):
    return tf.maximum(x - 110, 0)
