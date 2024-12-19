import numpy as np

class AmericanOptionUtils:
    def __init__(self, initial=100, T=0.25, volatility=0.3, periods=15, r=0.02, div=0.01, strike=110):
        self.initial = initial
        self.T = T
        self.volatility = volatility
        self.periods = periods
        self.r = r
        self.div = div
        self.strike = strike
        
        self.up = np.exp(self.volatility * np.sqrt(self.T / self.periods))
        self.down = 1 / self.up
        self.q1 = (np.exp((self.r - self.div) * self.T / self.periods) - self.down) / (self.up - self.down)
        self.q2 = 1 - self.q1
        self.size = self.periods + 1

    def generate_lattice(self):
        lattice = np.zeros((self.size, self.size))
        lattice[0, 0] = self.initial
        for j in range(1, self.size):
            for i in range(j + 1):
                if i < j:
                    lattice[i, j] = self.down * lattice[i, j - 1]
                elif i == j:
                    lattice[i, j] = self.up * lattice[i - 1, j - 1]
        return lattice

    def american_option_price(self, option_type):
        lattice = self.generate_lattice()
        option_matrix = np.zeros_like(lattice)

        if option_type == 'call':
            payoff = np.maximum(lattice[:, -1] - self.strike, 0)
        elif option_type == 'put':
            payoff = np.maximum(self.strike - lattice[:, -1], 0)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        option_matrix[:, -1] = payoff
        for j in range(self.periods - 1, -1, -1):
            for i in range(j + 1):
                continuation_value = (self.q1 * option_matrix[i + 1, j + 1] + self.q2 * option_matrix[i, j + 1]) \
                                     / np.exp(self.r * self.T / self.periods)
                intrinsic_value = 0
                if option_type == 'call':
                    intrinsic_value = max(lattice[i, j] - self.strike, 0)
                elif option_type == 'put':
                    intrinsic_value = max(self.strike - lattice[i, j], 0)

                option_matrix[i, j] = max(continuation_value, intrinsic_value)

        return option_matrix[0, 0]
