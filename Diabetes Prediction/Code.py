import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# PIMA Diabetes Dataset
diabetes_data = pd.read_csv("/home/negarmaghsoudi/Projects/PythonProjects/Diabetes Prediction/diabetes.csv")
print(diabetes_data.head)
print(diabetes_data.shape)
print(diabetes_data.describe())
print(diabetes_data.isnull().sum())
print(diabetes_data['Outcome'].value_counts())
print(diabetes_data.groupby('Outcome').mean())

X = diabetes_data.drop(columns = 'Outcome', axis = 1)
Y = diabetes_data['Outcome']

# Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_standard = scaler.transform(X_train)
X_test_standard = scaler.transform(X_test)

X_train, X_test = X_train_standard, X_test_standard

# Train the model
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_acc = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_acc = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on training data is : ', training_data_acc, ' , and Accuracy on test data is : ', test_data_acc)
#print("Accuracy on training data is: %.2f , and Accuracy on test data is: %.2f" % (training_data_acc, test_data_acc))

# Predict new data point's label
input_data = (6, 148, 72, 35, 0, 33.6, 0.627, 50)
input_data_arr = np.asarray(input_data)
input_data_reshape = input_data_arr.reshape(1,-1)
input_data_standard = scaler.transform(input_data_reshape)
predicted_class = classifier.predict(input_data_standard)
print(predicted_class)

if(predicted_class[0] == 1):
    print("Diabetic")
else:
    print("Non diabetic")