import numpy as np

def BS_CALL(S, K, T, r, sigma):
    from scipy.stats import norm
    N = norm.cdf
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r * T) * N(d2)

# Additional utility functions can be added here.