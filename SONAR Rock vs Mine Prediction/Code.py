import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv("/home/negarmaghsoudi/Projects/PythonProjects/SONAR Rock vs Mine Prediction/sonar data.csv", header = None)
print(sonar_data.head())
print(sonar_data.shape)
print(sonar_data.isnull().sum())
print(sonar_data.describe())
print(sonar_data[60].value_counts()) # M = 111, R = 97 -> almost the same number of samples in each class
print(sonar_data.groupby(60).mean())

X = sonar_data.drop(columns = 60, axis = 1)
Y = sonar_data[60]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)
print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()
model.fit(X_train, Y_train)

# Training data accuracy
X_train_pred = model.predict(X_train)
print(accuracy_score(X_train_pred, Y_train))

# Test data accuracy
X_test_pred = model.predict(X_test)
print(accuracy_score(X_test_pred, Y_test))

# Predict R --> Rock object
input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
input_arr = np.asarray(input_data)
input_reshape = input_arr.reshape(1,-1)
input_pred = model.predict(input_reshape)

if(input_pred[0] == "R"):
    print("The object is a Rock")
else:
    print("The object is a Mine")
