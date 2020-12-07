# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv('Data.csv')
dataset_copy = dataset.copy()
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,  -1].values

# Preprocessing Missing Values - Impute
from sklearn.impute import SimpleImputer
missingValues = SimpleImputer(missing_values=np.nan, strategy='mean')
missingValues = missingValues.fit(x[:, 1:])
x[:, 1:] = missingValues.transform(x[:, 1:])

# Encoding Categorical Data - Dummy [Example]
# dfDummy = pd.get_dummies(dataset_copy, columns=['Country'], prefix='', prefix_sep='')
# print(dfDummy)

# Encode Categorical Data - Label Encoder [Example]
# from sklearn.preprocessing import LabelEncoder
# encodeCountry = LabelEncoder()
# x[:, 0] = encodeCountry.fit_transform(x[:, 0])

# Encode Independent Categorical Data - One-Hot Encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), [0])], remainder='passthrough')
# x = ct.fit_transform(x)
x = np.array(ct.fit_transform(x), dtype=np.float)

# Encode Dependent Categorical Data - Label Encoder
from sklearn.preprocessing import LabelEncoder
encodePurchase = LabelEncoder()
y = encodePurchase.fit_transform(y)

# Splitting the dataset into Train Set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# print(dataset)
print(pd.DataFrame(x))
# print(y)

