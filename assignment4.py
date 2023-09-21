import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv('assignment-4/winequality-red.csv')
print(df)
df.info()
print(df.isnull().sum())
print(df.describe())
print(df.quality.unique())

print(sns.barplot(x=df.quality.value_counts(), y=df.alcohol.value_counts()))
"""spliting"""
x = df.iloc[:, :11]
y = df.iloc[:, -1]
print(x.info())
print(y.info())
"""Train, Test, Split¶"""
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=32)

"""Model Training¶"""
model1 = KNeighborsClassifier(n_neighbors=3)
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))


"""LogisticRegression"""
model2 = LogisticRegression(max_iter=5000)
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))


print("KNN Classifier Accuracy:", accuracy_score(y_test, y_pred1)*100)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred2)*100)

sample_check = [[6.5, 0.6, 0.3, 2.2, 0.07, 15.0, 40.0, 0.996, 3.4, 0.6, 9.5],
                [8.0, 0.4, 0.4, 2.8, 0.085, 22.0, 55.0, 0.998, 3.2, 0.55, 11.2],
                [6.8, 0.55, 0.15, 2.4, 0.075, 25.0, 62.0, 0.9962, 3.1, 0.75, 9.0],
                [7.5, 0.45, 0.35, 2.5, 0.09, 30.0, 70.0, 0.9978, 3.5, 0.6, 11.5],
                [7.0, 0.5, 0.2, 2.5, 0.08, 20.0, 60.0, 0.997, 3.3, 0.7, 10.0]
                ]

for i in sample_check:
    x = model2.predict([i])
    if (x >= 6):
        print(x, "--> Good")
    elif (x < 6):
        print(x, "--> Not Good")
