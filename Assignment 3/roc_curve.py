# -------------------------------------------------------------------------
# AUTHOR: Jahin Mahbub
# FILENAME: roc_curve.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 50 minutes
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

df = pd.read_csv('cheat_data.csv', sep=',', header=0)

X = []
y = []

for _, row in df.iterrows():
    refund = 1 if row['Refund'] == 'Yes' else 0
    single = 1 if row['Marital Status'] == 'Single' else 0
    divorced = 1 if row['Marital Status'] == 'Divorced' else 0
    married = 1 if row['Marital Status'] == 'Married' else 0
    income = float(row['Taxable Income'].lower().replace('k','').strip())
    X.append([refund, single, divorced, married, income])
    y.append(1 if row['Cheat'] == 'Yes' else 0)


trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3)

ns_probs = [0 for _ in range(len(testy))]

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()