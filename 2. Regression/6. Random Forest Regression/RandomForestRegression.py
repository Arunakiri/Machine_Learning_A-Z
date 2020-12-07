import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1:].values

# Fit Decision Tree Regression into the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, criterion='mse', random_state=0)
regressor.fit(x, y)

# Predict the Decision Tree Regression Results
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualizing the Decision Tree Regression Results (in Higher Dimension)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Random Forest)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
