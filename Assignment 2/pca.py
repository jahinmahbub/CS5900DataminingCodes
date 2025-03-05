# -------------------------------------------------------------------------
# AUTHOR: Jahin Mahbub
# FILENAME: pca.py
# SPECIFICATION: (pca.py) that will apply PCA multiple times on the
# heart_disease_dataset.csv, each time removing a single and distinct feature and printing the
# corresponding variance explained by PC1. Finally, find and print the maximal PC1 variance observed
# during the 10 iterations.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: A couple of hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
df = pd.read_csv("heart_disease_dataset.csv")

#Create a training matrix without the target variable (Heart Diseas)
df_features = df.drop(columns=["Heart Disease"], errors='ignore')
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features
num_features = df_features.shape[1]

pc1_variances = []
removed_features = []

# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    reduced_data = np.delete(scaled_data, i, axis=1)
    pca = PCA()
    pca.fit(reduced_data)
    pc1_variances.append(pca.explained_variance_ratio_[0])
    removed_features.append(df_features.columns[i])

# Find the maximum PC1 variance
max_pc1_variance = max(pc1_variances)
best_feature_removed = removed_features[pc1_variances.index(max_pc1_variance)]

#Print results
print(f"Highest PC1 variance found: {max_pc1_variance:.6f} when removing {best_feature_removed}")





