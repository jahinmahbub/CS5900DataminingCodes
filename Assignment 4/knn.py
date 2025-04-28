#-------------------------------------------------------------------------
# AUTHOR: Jahin Mahbub
# FILENAME: knn.py
# SPECIFICATION: K nearest Neighbor
# FOR: CS 5990- Assignment #4
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

def discretize(y_values):
    return np.array([min(classes, key=lambda x: abs(x - v)) for v in y_values])

#reading the training data
df_training = pd.read_csv('weather_training.csv', sep=',', header=0)
X_training = np.array(df_training.iloc[:, 1:-1]).astype('f')
y_training = discretize(np.array(df_training.iloc[:, -1]).astype('f'))

#reading the test data
df_test = pd.read_csv('weather_test.csv', sep=',', header=0)
X_test = np.array(df_test.iloc[:, 1:-1]).astype('f')
y_test = discretize(np.array(df_test.iloc[:, -1]).astype('f'))

#loop over the hyperparameter values (k, p, and w) ok KNN
highest_accuracy = 0
for k in k_values:
    for p in p_values:
        for w in w_values:

            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            correct_predictions = 0
            total_predictions = 0

            for (x_testSample, y_testSample) in zip(X_test, y_test):
                y_predicted = clf.predict([x_testSample])[0]
                perc_diff = 100 * abs(y_predicted - y_testSample) / abs(y_testSample)
                if perc_diff <= 15:
                    correct_predictions += 1
                total_predictions += 1

            accuracy = correct_predictions / total_predictions

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {round(highest_accuracy,2)}, Parameters: k={k}, p={p}, weight={w}")





