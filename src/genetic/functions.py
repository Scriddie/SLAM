import numpy as np


def rosenbrock(pos, a=1, b=100):
    x1, x2 = pos
    return (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2


def rosenbrock_grad(pos, a=1, b=100):
    x, y = pos
    x_grad = 2.0 * (a - x) * (-1.0) + 2.0 * b * (y - np.square(x)) * (-2.0 * x)
    y_grad = 2.0*b*(y - np.square(x))
    return np.array([x_grad, y_grad])


def rastrigin(pos, a=10):
    x1, x2 = pos
    X = np.stack((x1, x2), axis=-1)
    res = X ** 2 - a * np.cos(2 * np.pi * X)
    return 2 * a + np.sum(res, axis=-1)


def rastrigin_grad(pos, A=10):
    x, y = pos
    x_grad = A * len(pos) + 2*x - 10*np.sin(2*np.pi*x) * 2*np.pi
    y_grad = A * len(pos) + 2*y - 10*np.sin(2*np.pi*y) * 2*np.pi
    return np.array([x_grad, y_grad])


def distance_mse(x_points, y_points, x_true, y_true):
    # First calculate the Euclidian distance then take the mse
    sum = 0
    for i in range(len(x_points)):
        # Calculate the euclidian distance
        dist = np.sqrt((x_points[i] - x_true)**2 + (y_points[i] - y_true)**2)

        # The true distance is 0
        dist_true = 0
        sum += (dist - dist_true)**2

    mse = sum/float(len(x_points))
    return mse