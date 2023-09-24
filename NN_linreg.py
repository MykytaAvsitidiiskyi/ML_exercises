from typing import Tuple
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1337)
torch.manual_seed(314)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


if __name__ == "__main__":
    # Get available device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device = }")

    # Check PyTorch version
    print(f"Using {torch.__version__ = }")

    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    x, y_true, y = torch.tensor(x), torch.tensor(y_true), torch.tensor(y)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    # Create train/test split
    train_split = int(0.8 * len(x))
    X_train, y_train = x[:train_split], y[:train_split]
    X_test, y_test = x[train_split:], y[train_split:]

    # Create an instance of the model
    model_0 = LinearRegressionModel()

    # Check the initial weights before training
    print("Initial Weights:")
    print("Weight:", model_0.weights.item())
    print("Bias:", model_0.bias.item())

    # Create the loss function
    loss_fn = nn.L1Loss()

    # Create the optimizer
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    # Set the number of epochs
    epochs = 100

    # Create lists to store the loss values for train and test data
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        ### Training
        model_0.train()
        y_pred = model_0(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Testing
        model_0.eval()
        with torch.no_grad():
            test_pred = model_0(X_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float))

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

    # Plot the loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MAE Loss")
    plt.legend()
    plt.savefig("loss_curve.png")

    # Check the final weights after training
    print("Final Weights:")
    print("Weight:", model_0.weights.item())
    print("Bias:", model_0.bias.item())

    # Compare with original weights
    original_weights = 2.0
    original_bias = 3.5
    print("Comparison with Original Weights:")
    print("Weight (Before):", original_weights)
    print("Weight (After):", model_0.weights.item())
    print("Bias (Before):", original_bias)
    print("Bias (After):", model_0.bias.item())

    # Plot results (predicted line along with dataset data)
    model_0.eval()
    with torch.no_grad():
        y_pred_all = model_0(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y_true, label="True")
    plt.plot(x, y_pred_all, label="Predicted", linestyle="--")
    plt.xlabel("Input (x)")
    plt.ylabel("Output (y)")
    plt.legend()
    plt.savefig("results.png")
