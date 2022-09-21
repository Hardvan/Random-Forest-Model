import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Getting Data
loans = pd.read_csv("loan_data.csv")
print(loans.head())
print()
print(loans.describe())
print()
print(loans.info())
print()

# EDA
plt.figure(figsize=(10,6))
loans[loans["credit.policy"] == 1]["fico"].hist(bins = 30, color = "blue",
                                               label = "Credit Policy 1")
loans[loans["credit.policy"] == 0]["fico"].hist(bins = 30, color = "red",
                                               label = "Credit Policy 2")
plt.legend()
plt.xlabel("FICO")

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
plt.legend()
plt.xlabel('FICO')

plt.figure(figsize=(10,6))
sns.countplot(x = loans["purpose"], data = loans, hue = "not.fully.paid", palette="viridis")
plt.tight_layout()

sns.jointplot(x = "fico", y = "int.rate", data = loans, color = "purple")

plt.figure(figsize=(10,7))
sns.lmplot(x = "fico", y = "int.rate", data = loans, hue = "credit.policy",
          col = "not.fully.paid", palette = "magma")

# Data Preprocessing
print(loans.info())
# purpose column as categorical

final_data = pd.get_dummies(loans, columns = ["purpose"], drop_first = True)

print(final_data.head())

# Training Model
from sklearn.model_selection import train_test_split
X = final_data.drop("not.fully.paid", axis = 1)
y = final_data["not.fully.paid"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 42)

# Single Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Predictions & Evaluation
dtree_pred = dtree.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print("Decision Tree Model Report")
print(confusion_matrix(y_test, dtree_pred))
print()
print(classification_report(y_test, dtree_pred))
print()

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 600)
rfc.fit(X_train, y_train)

# Predictions & Evaluation
rfc_pred = rfc.predict(X_test)

print("Random Forest Model Report")
print(confusion_matrix(y_test, rfc_pred))
print()
print(classification_report(y_test, rfc_pred))
print()
















