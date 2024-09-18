from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

df = pd.read_csv('health.csv')
X = df.drop('death rate', axis=1)
Y_train = df['death rate']
X_train = StandardScaler().fit_transform(X)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_train)


mse = mean_squared_error(y_pred, Y_train)
print(model.coef_, model.intercept_)
print(mse)
