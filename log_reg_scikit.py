import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('social_network.csv')
X = df.drop(['User ID', 'Purchased', 'Gender'], axis=1)
Y = df['Purchased']
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

lr_model = LogisticRegression()
lr_model.fit(X_scaled, Y)
print(lr_model.coef_, lr_model.intercept_)
