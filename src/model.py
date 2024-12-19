import tensorflow as tf
import numpy as np

class DeepBSDE(tf.keras.Model):
    def __init__(self, f, x, g, sigma, mu, t0=0.0, t1=1.0, dim=100, time_steps=20, learning_rate=1e-2, num_hidden_layers=2, num_neurons=200, **kwargs):
        super().__init__(**kwargs)
        self.t0 = t0
        self.t1 = t1
        self.x = x
        self.f = f
        self.g = g
        self.sigma = sigma
        self.mu = mu
        self.N = time_steps
        self.dim = dim
        self.dt = (t1 - t0) / self.N
        self.sqrt_dt = np.sqrt(self.dt)
        self.t_space = np.linspace(self.t0, self.t1, self.N + 1)[:-1]
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, epsilon=1e-8)
        self.u0 = tf.Variable(np.random.uniform(.3, .5, size=(1)).astype('float32'))
        self.gradu0 = tf.Variable(np.random.uniform(-1e-1, 1e-1, size=(1, dim)).astype('float32'))
        self.gradui = self._build_grad_network(num_hidden_layers, num_neurons, dim)

    def _build_grad_network(self, num_hidden_layers, num_neurons, dim):
        gradui = []
        _dense = lambda dim: tf.keras.layers.Dense(units=dim, activation=None, use_bias=False)
        _bn = lambda: tf.keras.layers.BatchNormalization(momentum=.99, epsilon=1e-6, beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1), gamma_initializer=tf.random_uniform_initializer(0.1, 0.5))
        for _ in range(self.N - 1):
            this_grad = tf.keras.Sequential()
            this_grad.add(tf.keras.layers.Input(dim))
            this_grad.add(_bn())
            for _ in range(num_hidden_layers):
                this_grad.add(_dense(num_neurons))
                this_grad.add(_bn())
                this_grad.add(tf.keras.layers.ReLU())
            this_grad.add(_dense(dim))
            this_grad.add(_bn())
            gradui.append(this_grad)
        return gradui

    def draw_X_and_dW(self, num_sample):
        dW = np.random.normal(loc=0.0, scale=self.sqrt_dt, size=(num_sample, self.dim, self.N)).astype('float32')
        X = np.zeros((num_sample, self.dim, self.N + 1), dtype='float32')
        X[:, :, 0] = self.x
        for i in range(self.N):
            t = self.t_space[i]
            X[:, :, i + 1] = X[:, :, i] + self.mu(t, X[:, :, i]) * self.dt + tf.reduce_sum(self.sigma(t, X[:, :, i]) * dW[:, :, i], axis=1, keepdims=True)
        return X, dW

    @tf.function
    def call(self, inp, training=False):
        X, dW = inp
        num_sample = X.shape[0]
        e_num_sample = tf.ones(shape=[num_sample, 1], dtype='float32')
        y = e_num_sample * self.u0
        z = e_num_sample * self.gradu0
        for i in range(self.N - 1):
            t = self.t_space[i]
            eta1 = -self.f(t, X[:, :, i], y, z) * self.dt
            eta2 = tf.reduce_sum(z * dW[:, :, i], axis=1, keepdims=True)
            y = y + eta1 + eta2
            z = self.gradui[i](X[:, :, i + 1], training)
        eta1 = -self.f(self.t_space[self.N - 1], X[:, :, self.N - 1], y, z) * self.dt
        eta2 = tf.reduce_sum(z * dW[:, :, self.N - 1], axis=1, keepdims=True)
        y = y + eta1 + eta2
        return y

    def loss_fn(self, inputs, training=False):
        X, _ = inputs
        y_pred = self.call(inputs, training)
        y = self.g(X[:, :, -1])
        y_diff = y - y_pred
        loss = tf.reduce_mean(tf.square(y_diff))
        return loss

    @tf.function
    def train(self, inp):
        loss, grad = self.grad(inp, training=True)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

    @tf.function
    def grad(self, inputs, training=False):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.trainable_variables)
        return loss, grad
