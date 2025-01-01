from model import DeepBSDE
from training import train_model
from utils import f, g, mu, sigma
from american_option import AmericanOptionUtils

# Initialize DeepBSDE model
model = DeepBSDE(f=f, mu=mu, sigma=sigma, g=g, x=100, dim=1)

# Train the model
train_model(model, epochs=3000, batch_size=30)

# Print results
inp = model.draw_X_and_dW(30)
print(f"Estimated initial value (u0): {model.u0.numpy()}")
print(f"Loss: {model.loss_fn(inp).numpy()}")

# Calculate American option prices
option_utils = AmericanOptionUtils(initial=100, T=1, volatility=0.2, periods=50, r=0.03, div=0.0, strike=110)
american_call_price = option_utils.american_option_price('call')
american_put_price = option_utils.american_option_price('put')

print(f"American Call Option Price: {american_call_price}")
print(f"American Put Option Price: {american_put_price}")
