import numpy as np

X = np.arange(40)
# print(X)
data = np.array_split(X, 4)
# print(data)

data1 = np.concatenate(data[:2]+data[3:])
print(np.array([data1]))

# print(data[:2] + data[3:])
# data_split = np.concatenate(data[:2] + data[3:])
# print(data_split)
