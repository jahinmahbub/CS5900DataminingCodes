# -------------------------------------------------------------------------
# AUTHOR: Jahin Mahbub
# FILENAME: decision_tree.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 50 minutes
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    for row in data_training:
        refund = 1 if row[0] == 'Yes' else 0
        single = 1 if row[1] == 'Single' else 0
        divorced = 1 if row[1] == 'Divorced' else 0
        married = 1 if row[1] == 'Married' else 0
        income = float(row[2].lower().replace('k', '').strip())
        X.append([refund, single, divorced, married, income])
        Y.append(1 if row[3] == 'Yes' else 2)

    total_accuracy = 0

    for i in range(10):

        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()

        df_test = pd.read_csv('cheat_test.csv', sep=',', header=0)
        data_test = np.array(df_test.values)[:,1:]

        correct = 0
        total = 0

        for data in data_test:
            refund = 1 if data[0] == 'Yes' else 0
            single = 1 if data[1] == 'Single' else 0
            divorced = 1 if data[1] == 'Divorced' else 0
            married = 1 if data[1] == 'Married' else 0
            income = float(row[2].lower().replace('k', '').strip())
            class_predicted = clf.predict([[refund, single, divorced, married, income]])[0]
            true_label = 1 if data[3] == 'Yes' else 2
            if class_predicted == true_label:
                correct += 1
            total += 1

        accuracy = correct / total
        total_accuracy += accuracy

    final_accuracy = total_accuracy / 10
    print('final accuracy when training on', ds + ':', round(final_accuracy, 2))