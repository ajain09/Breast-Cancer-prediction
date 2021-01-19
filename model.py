from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv(
    "https://raw.githubusercontent.com/ajain09/Coursera/main/data.csv")
df.drop(columns=['Unnamed: 32', 'id'], axis=1, inplace=True)
features = df.drop(columns=['diagnosis'])
y = df['diagnosis']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(features)
pca = PCA()
X = pca.fit_transform(x_scaled)
X.shape
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.30, random_state=42)
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)
#y_pred = lr.predict(X_test)

pickle.dump(lr, open('model.pkl', 'wb'))
pred_model = pickle.load(open('model.pkl', 'rb'))
y_pred = pred_model.predict(X_test)
