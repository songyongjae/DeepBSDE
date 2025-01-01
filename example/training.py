def train_model(model, epochs=1000, batch_size=30):
    for epoch in range(epochs):
        inp = model.draw_X_and_dW(batch_size)
        loss, grad = model.grad(inp, training=True)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")
