from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(1337)


def get_data(nsamples: int = 100) -> Tuple[np.array, np.array]:
    x = np.linspace(0, 10, nsamples)
    y = 2 * x + 3.5
    return (x, y)


def add_noise(y: np.array) -> np.array:
    noise = np.random.normal(size=y.size)
    return y + noise


def mse_regression(guess: np.array, x: np.array, y: np.array) -> float:
    """MSE Minimization Regression"""
    m = guess[0]
    b = guess[1]
    # Predictions
    y_hat = m * x + b
    # Get loss MSE
    mse = (np.square(y - y_hat)).mean()
    return mse


if __name__ == "__main__":
    # Getting data
    x, y_true = get_data()
    y = add_noise(y_true)

    # Plot and investigate data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.legend(loc="best")
    plt.savefig("data.png")

    # Initial guess of the parameters: [2, 2] (m, b).
    # It doesn’t have to be accurate but simply reasonable.
    initial_guess = np.array([5, -3])
    mse_values = [] #Створимо порожній список для зберігання значень mse
    mse_values.append(mse_regression(initial_guess, x, y)) #Додаємо в список значення mse для початкового наближення

    # Maximizing the probability for point to be from the distribution
    results = minimize(
        mse_regression,
        initial_guess,
        args=(x, y,),
        method="Nelder-Mead",
        options={"disp": True},
        callback = lambda xk: #Ось тут додамо callback функцію, котра буде викликатися на кожній ітерації
        mse_values.append(mse_regression(xk, x, y))
    )

    print(results)
    print("Parameters: ", results.x)
    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6)) #Малюємо графік mse
    ax.plot(range(len(mse_values)), mse_values, "g-", label="MSE")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.legend(loc="best")
    plt.savefig("mse_plot.png")

    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = results.x[0] * xx + results.x[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o", label="data")
    ax.plot(x, y_true, "b-", label="True")
    ax.plot(xx, yy, "r--.", label="MLE")
    ax.legend(loc="best")

    plt.savefig("mle_regression.png")
