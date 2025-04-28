#-------------------------------------------------------------------------
# AUTHOR: Jahin Mahbub
# FILENAME: naive_bayes.py
# SPECIFICATION: naive bayes calculation
# FOR: CS 5990- Assignment #4
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
df_training = pd.read_csv('weather_training.csv')
X_training = np.array(df_training.iloc[:, 1:-1]).astype('f')
y_training = np.array(df_training.iloc[:, -1]).astype('f')

#update the training class values according to the discretization (11 values only)
for i in range(len(y_training)):
    y_training[i] = min(classes, key=lambda x: abs(x - y_training[i]))

#reading the test data
df_test = pd.read_csv('weather_test.csv')
X_test = np.array(df_test.iloc[:, 1:-1]).astype('f')
y_test = np.array(df_test.iloc[:, -1]).astype('f')

#update the test class values according to the discretization (11 values only)
for i in range(len(y_test)):
    y_test[i] = min(classes, key=lambda x: abs(x - y_test[i]))

#loop over the hyperparameter value (s)
highest_accuracy = 0

for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    correct = 0
    total = 0

    for (x_testSample, y_testSample) in zip(X_test, y_test):
        predicted = clf.predict([x_testSample])[0]
        if y_testSample != 0:
            percentage_diff = 100 * abs(predicted - y_testSample) / abs(y_testSample)
            if percentage_diff <= 15:
                correct += 1
            total += 1

    accuracy = correct / total

    #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        print(f"Highest Naive Bayes accuracy so far: {highest_accuracy}, Parameters: s={s}")



