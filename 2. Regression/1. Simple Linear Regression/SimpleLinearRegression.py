import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, 0].values.reshape(-1, 1)
y = dataset.iloc[:, 1].values.reshape(-1, 1)

# Splitting dataset into Train and Test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regressing into the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Train and Test Set Results
y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

# Visualizing the Training Set Results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, y_train_pred, color='blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test Set Results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, y_train_pred, color='blue')
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

