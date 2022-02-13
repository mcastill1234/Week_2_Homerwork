import numpy as np
from HyperplaneProcedures import cv


def perceptron(data, labels):
    dim, num_samples = data.shape
    theta = np.zeros((dim, 1))
    theta_0 = np.zeros((1, 1))
    mistakes = 0
    for tao in range(100):
        changed = False
        for i in range(num_samples):
            y_i = labels[0, i]
            x_i = data[:, i]
            current_guess = y_i * (np.dot(theta.T, x_i) + theta_0)
            if current_guess <= 0:
                mistakes += 1
                theta = theta + cv(y_i * x_i)
                theta_0 = theta_0 + y_i
                changed = True
                print("New theta = ", theta.T)
                print("New theta_0 = ", theta_0)
        if not changed:
            break
    print("Number of mistakes: ", mistakes)
    print()
    return theta, mistakes


# Test case for exercises 3.1 and 3.2
X = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]])
y = np.array([[-1, -1, -1, -1, -1, -1, -1, 1]])
perceptron(X, y)
