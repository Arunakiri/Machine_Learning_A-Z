import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode Categorical Data - One-Hot Encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)

# Avoiding Dummy Variable Trap
x = x[:, 1:]

# Building the Optimal Model - Backward Elimination
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)

# Finding the P-value with OLS - Ordinary Least Square
import statsmodels.api as sm

x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# print(regressor_OLS.summary())

# Remove the Variables with P-value > Significance Level
"""
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())
"""

x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
# print(regressor_OLS.summary())

# Splitting dataset into Train and Test Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression into Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting Training Set Results
y_train_pred = regressor.predict(x_train)
print(np.vstack((y_train, y_train_pred)).T)

print('\n')

# Predicting Test Set Results
y_test_pred = regressor.predict(x_test)
print(np.vstack((y_test, y_test_pred)).T)












