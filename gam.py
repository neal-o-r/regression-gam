import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from regression_gam import RegressionGAMGCV


def f(x):

    return x ** 2 - x


def get_data(n_points):

    x = np.linspace(-2, 2, n_points)
    x.sort()
    u = 0.5
    y = f(x) + np.random.randn(n_points) * u

    return x, y


if __name__ == "__main__":

    x, y = get_data(50)

    reg = RegressionGAMGCV()
    reg.fit(x, y)

    y_p = reg.predict(x)

    plt.plot(x, y, ".")
    plt.plot(x, y_p, label="Prediction")
    plt.plot(x, f(x), label="True")
    plt.legend()
    plt.show()
