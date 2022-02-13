import numpy as np
from HyperplaneProcedures import cv
from HyperplaneProcedures import normalize


def perceptron_through_origin(data, labels):
    theta = np.zeros((2, 1))
    mistakes = 0
    for tao in range(20):
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


# Test case for exercises 1.1a and 1.1b
X = np.array([[1, 0, -1.5], [-1, 1, -1]])
y = np.array([[1, -1, 1]])
print("Solutions to 1.1a) and 1.1b): ")
th, th_0 = perceptron_through_origin(X, y)
print("normalized theta: ", normalize(th))


# Test case for exercises 1.1c and 1.1d
X = np.array([[0, -1.5, 1], [1, -1, -1]])
y = np.array([[-1, 1, 1]])
print("Solutions to 1.1c) and 1.1d): ")
perceptron_through_origin(X, y)
print("normalized theta: ", normalize(th))


# Test case for exercise 1.2a
X = np.array([[1, 0, -10], [-1, 1, -1]])
y = np.array([[1, -1, 1]])
print("Solution to 1.2a): ")
perceptron_through_origin(X, y)
print("normalized theta: ", normalize(th))

# Test case for exercise 1.2b
X = np.array([[0, -10, 1], [1, -1, -1]])
y = np.array([[-1, 1, 1]])
print("Solution to 1.2b): ")
perceptron_through_origin(X, y)
print("normalized theta: ", normalize(th))

