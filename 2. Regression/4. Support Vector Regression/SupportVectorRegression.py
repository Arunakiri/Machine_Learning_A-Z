import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Fit Support Vector Regression into the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

# Predict the SVR Result
# y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]))))
y_pred = regressor.predict(sc_x.transform([[6.5]]))
print(y_pred)

# Visualizing the SVR Results
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
