import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv("possum.csv")
df = df.dropna()
X = df.drop(["case", "site", "Pop", "sex"], axis=1) 
y = df["sex"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=11)
rf_model.fit(X_train, y_train)
pickle.dump(rf_model, open("possum_rf_model.pkl","wb"))
