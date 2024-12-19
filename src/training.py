import tensorflow as tf

def train_model(model, num_iterations=1000):
    for i in range(num_iterations):
        inp = model.draw_X_and_dW(30)
        model.train(inp)
        print(f"Iteration {i}: u0={model.u0}, Loss={model.loss_fn(inp)}")
