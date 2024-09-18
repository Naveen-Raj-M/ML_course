import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('social_network.csv')
X = df.drop(['User ID', 'Purchased', 'Gender'], axis=1)
Y = df['Purchased']
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def compute_cost(X, Y, W, b, lambda_=1):
    m, n = X.shape
    cost = 0
    for i in range(m):
        z_i = np.dot(X[i], W) + b
        f_wb_i = sigmoid(z_i)
        cost += (-Y[i] * np.log(f_wb_i)) - ((1 - Y[i]) * np.log(1 - f_wb_i))

    cost = cost/m
    reg_cost = 0
    for j in range(n):
        reg_cost += (W[j] ** 2)

    reg_cost = (lambda_ * reg_cost)/ (2 * m)

    total_cost = cost + reg_cost
    return total_cost

def compute_gradient(X, Y, W, b, lambda_=1):
    m, n = X.shape
    dW = np.zeros((n,))
    db = 0.0
    for i in range(m):
        z_i = np.dot(X[i], W) + b
        f_wb_i = sigmoid(z_i)
        err_i = f_wb_i - Y[i]

        for j in range(n):
            dW[j] += err_i * X[i,j]

        db += err_i

    for j in range(n):
        dW[j] = dW[j] + ((lambda_ / m) * W[j])

    dW = dW / m
    db = db / m
    return dW, db

def gradient_descent(X, Y, W_init, b_init, itr, alpha=0.01, lambda_=1):
    j_history = []
    W = W_init
    b = b_init

    for i in range(itr):
        dW, db = compute_gradient(X, Y, W, b, lambda_)

        W = W - (alpha*dW)
        b = b - (alpha*db)

        j_history.append(compute_cost(X, Y, W, b, lambda_))

    return W, b, j_history

W_init = np.zeros((X.shape[1]))
b_init = 0.
alpha = 1e-2
itr = 20000
lambda_ = 1
W_final, b_final, j_history = gradient_descent(X_scaled, Y, W_init, b_init, itr, alpha, lambda_)

print(W_final, b_final)
print(j_history[-1])
