import numpy as np
from matplotlib import pyplot as plt

# Random Data Generation
# Monkey Data as Attributes height with respect to weight
monkey_data = np.random.multivariate_normal([1, 2], [[1, 0.5], [0.5, 1]], 1000)

# Tiger Data as Attributes height with respect to weight
tiger_data = np.random.multivariate_normal([4, 4], [[1, 0.5], [0.5, 1]], 1000)


##distance between two point function
# Euclidian Distance

def dist(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum())



# KNN Function
def knn(x_train, y_train, query_point, k=5):
    dist_vals = []
    m = x_train.shape[0]
    for ix in range(m):
        d = dist(query_point, x_train[ix])
        dist_vals.append((d, y_train[ix]))
    dist_vals = sorted(dist_vals)
    dist_vals = dist_vals[:k]

    y = np.array(dist_vals)
    t = np.unique(y[:, 1], return_counts=True)
    index = t[1].argmax()

    # Prediction for
    # 0 = Monkey
    # 1=Tiger

    prediction = t[0][index]

    print(prediction)



data = np.zeros((2000, 3))
data[:1000, :-1] = monkey_data
data[1000:, :-1] = tiger_data
data[:1000, -1] = 0
data[1000:, -1] = 1

# Training

x_train = data[:, :-1]
y_train = data[:, -1]


plt.scatter(monkey_data[:, 0], monkey_data[:, 1], label='Monkey')
plt.scatter(tiger_data[:, 0], tiger_data[:, 1], alpha=0.5, label='Tiger')
# taking an arbitary Query Point
query_point = [3, 2]
plt.scatter(query_point[0], query_point[1], c='k', label='Query')
plt.legend()
plt.show()

# Calling
knn(x_train, y_train, query_point)


