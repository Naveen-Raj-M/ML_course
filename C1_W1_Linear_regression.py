import numpy as np
import matplotlib.pyplot as plt

#training data
x_train = np.array([1, 2]) # in 1000 sqft
y_train = np.array([300, 500]) # in 1000 dollars

#to compute cost
def compute_cost(x, y, w, b):
    training_set = x.shape[0]
    cost = 0

    for i in range(training_set):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2

    total_cost = (1 / (2*training_set)) * cost
    return total_cost

# to compute gradient
def compute_gradient(x, y, w, b):
    training_set = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(training_set):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / training_set
    dj_db = dj_db / training_set

    return dj_dw, dj_db

# to compute gradient descent
def gradient_descent(x, y, w_in, b_in, alpha, num_itr, compute_cost, compute_gradient):
    j_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_itr):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        j_history.append(compute_cost(x, y, w, b))
        p_history.append([w, b])

    return w, b, j_history, p_history

w_init = 0
b_init = 0
itr = 10000
alpha = 1e-2
w_final, b_final, j_hist, p_hist = gradient_descent(x_train, y_train,
                                                          w_init, b_init,
                                                          alpha, itr,
                                                          compute_cost,
                                                          compute_gradient)
