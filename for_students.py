import random

import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
n=len(x_train)
x = x_train
#x_1 = np.pad(x, ((0, 0), (1, 0)), mode="constant", constant_values=(1, 1))
x_1=np.ones((n,2))
x_1[:,1] = x
x_t = x_1.T
y = np.matrix(y_train)
x_tx = x_t.dot(x_1)
x_txI = np.linalg.inv(x_tx)

theta_best = [0, 0]
theta_best = x_txI.dot(x_t)
theta_best = theta_best.dot(y.T)

# TODO: calculate error

MSE = (theta_best.T).dot(x_1.T)
MSE = MSE - y_train
MSE = (MSE).dot(MSE.T)
MSE = 1/n * MSE

print(MSE)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization

x_norm = (x_train - (x_train.mean(axis=0))) / (x_train.std(axis=0))
print(x_norm.mean())
y_norm = (y_train - (y_train.mean(axis=0))) / (y_train.std(axis=0))
print(y_norm.mean())
x_norm_test = (x_test - (x_train.mean(axis=0))) / (x_train.std(axis=0))
y_norm_test = (y_test - (y_train.mean(axis=0))) / (y_train.std(axis=0))


# TODO: calculate theta using Batch Gradient Descent
theta_best = np.random.rand(1,2)
#for i in range(100):
x=x_norm
x_1=np.ones((n,2))
x_1[:,1] = x
x1 = x_1.dot(theta_best.T)
x2 = np.subtract(y_norm, x1)
x3 = x_1.T.dot(x2)/len(x)
gradient = -2 * x3

n = 0.001
#theta_best = theta_best - n * gradient


# TODO: calculate error

# plot the regression line
x = np.linspace(min(x_norm_test), max(x_norm_test), 100)
y = float(theta_best[0,1]) + float(theta_best[0,1]) * x
plt.plot(x, y)
plt.scatter(x_norm_test, y_norm_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
