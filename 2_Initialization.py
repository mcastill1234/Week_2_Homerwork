import numpy as np
from HyperplaneProcedures import cv
from HyperplaneProcedures import normalize


def perceptron_through_origin(data, labels):
    theta = cv([1000, -1000])
    mistakes = 0
    for tao in range(2000):
        changed = False
        for i in range(3):
            y_i = labels[0, i]
            x_i = data[:, i]
            current_guess = y_i * np.dot(theta.T, x_i)
            if current_guess <= 0:
                mistakes += 1
                theta = theta + cv(y_i * x_i)
                changed = True
                print("New theta = ", theta.T)
        if not changed:
            break
    print("Number of mistakes: ", mistakes)
    print()
    return theta, mistakes


# Test case for exercises 2.1
X = np.array([[1, 0, -1.5], [-1, 1, -1]])
y = np.array([[1, -1, 1]])
print("Solutions to 2.1: ")
th, th_0 = perceptron_through_origin(X, y)
print("normalized theta: ", normalize(th))