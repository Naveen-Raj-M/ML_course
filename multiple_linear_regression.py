import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('health.csv')
X = df.drop('death rate', axis=1)
Y = df['death rate']
sc = StandardScaler()
X_train = sc.fit_transform(X)

def compute_cost(X, Y, W, b):
    m,n = X.shape
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i].reshape(1,n), W) + b
        cost += (f_wb_i - Y[i])**2

    cost = cost / (2 * m)
    return cost

def compute_gradient(X, Y, W, b):
    m, n = X.shape
    dW = np.zeros((n,1))
    db = 0.0
    for i in range(m):
        error = (np.dot(X[i].reshape(1,n), W.reshape(n,1)) + b) - Y[i]
        for j in range(n):
            dW[j] = dW[j].reshape(1, 1) + (error.reshape(1,1) * X[i,j].reshape(1,1))
        db += error
    dW = (dW / m).reshape(n,1)
    db = db / m
    return dW, db

def gradient_descent(X, Y, W, b, alpha, itr, compute_cost, compute_gradient):
    j_history = []

    for i in range(itr):
        dW, db = compute_gradient(X, Y, W, b)
        W = W - (alpha * dW)
        b = b - (alpha * db)

        j_history.append(compute_cost(X, Y, W, b))

    return W, b, j_history


W_init = np.zeros([(X_train.shape)[1],1])
b_init = 0
itr = 1000
alpha = 1e-2
W_final, b_final, j_hist = gradient_descent(X_train, Y,
                                                          W_init, b_init,
                                                          alpha, itr,
                                                          compute_cost,
                                                          compute_gradient)

print(W_final, b_final, j_hist[-1])