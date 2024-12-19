from training import trained_model
from black_scholes import BS_CALL

# Compare DeepBSDE model output with Black-Scholes pricing
if __name__ == "__main__":
    S = 100  # Current stock price
    K = 110  # Strike price
    T = 1.0  # Time to maturity
    r = 0.03  # Risk-free interest rate
    sigma = 0.2  # Volatility

    # Black-Scholes Price
    bs_price = BS_CALL(S, K, T, r, sigma)
    print(f"Black-Scholes Call Price: {bs_price}")

    # DeepBSDE Model Prediction
    inp = trained_model.draw_X_and_dW(1)
    model_price = trained_model.call(inp)
    print(f"DeepBSDE Predicted Price: {model_price.numpy()[0][0]}")