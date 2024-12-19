import tensorflow as tf
import numpy as np

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

class DeepBSDE(tf.keras.Model):
    def __init__(self, f, x, g, sigma, mu, t0=0.0, t1=1.0, dim=100, time_steps=20, learning_rate=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.t0, self.t1, self.x = t0, t1, x
        self.f, self.g, self.sigma, self.mu = f, g, sigma, mu
        self.N = time_steps
        self.dt = (t1 - t0) / self.N
        self.sqrt_dt = np.sqrt(self.dt)

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
        self.u0 = tf.Variable(np.random.uniform(.3, .5, size=(1)).astype(DTYPE))
        self.gradu0 = tf.Variable(np.random.uniform(-1e-1, 1e-1, size=(1, dim)).astype(DTYPE))

        self.gradui = []
        for _ in range(self.N - 1):
            this_grad = tf.keras.Sequential([
                tf.keras.layers.Dense(200, activation='relu') for _ in range(2)
            ])
            self.gradui.append(this_grad)

    def draw_X_and_dW(self, num_sample):
        dW = np.random.normal(scale=self.sqrt_dt, size=(num_sample, self.x.shape[1], self.N)).astype(DTYPE)
        X = np.zeros((num_sample, self.x.shape[1], self.N + 1), dtype=DTYPE)
        X[:, :, 0] = self.x
        for i in range(self.N):
            X[:, :, i + 1] = X[:, :, i] + self.mu(X[:, :, i]) * self.dt + self.sigma(X[:, :, i]) @ dW[:, :, i]
        return X, dW

    @tf.function
    def call(self, inputs):
        X, dW = inputs
        y = tf.ones((X.shape[0], 1), dtype=DTYPE) * self.u0
        z = tf.ones((X.shape[0], X.shape[1]), dtype=DTYPE) * self.gradu0
        for grad_fn in self.gradui:
            y += -self.f(X, y, z) * self.dt + tf.reduce_sum(z * dW, axis=1, keepdims=True)
            z = grad_fn(X)
        return y

    def loss_fn(self, inputs):
        X, _ = inputs
        y_pred = self.call(inputs)
        y_true = self.g(X[:, :, -1])
        return tf.reduce_mean(tf.square(y_true - y_pred))
