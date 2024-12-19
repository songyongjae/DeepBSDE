from utils import BS_CALL

# Example usage of Black-Scholes pricing function
if __name__ == "__main__":
    S = 100  # Current stock price
    K = 110  # Strike price
    T = 1.0  # Time to maturity
    r = 0.03  # Risk-free interest rate
    sigma = 0.2  # Volatility

    call_price = BS_CALL(S, K, T, r, sigma)
    print(f"Black-Scholes Call Price: {call_price}")
