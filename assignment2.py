from numpy import number
import pandas as py
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc_params

data = py.read_csv(
    "/Users/nani/Desktop/ai and ml/assignment 2/House Price India.csv")
# data.head()
# print(sns.jointplot(data))

# univatiate

#sns.barplot(x=data.Price.value_counts(), y=data.Price.value_counts())
# bivariate
sns.scatterplot(x=data.Price, y=data.Date)
# multivariant
#sns.pairplot(data)

# print(data.describe())
# print(data.median())
data.isnull().any()
# there are no null values if there are null values
data.fillna(data.mode())
