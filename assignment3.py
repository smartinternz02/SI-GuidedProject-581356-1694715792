import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('penguins_size.csv')

df.head()
# univatiate
# sns.displot(df.body_mass_g)

# sns.barplot(x=df.culmen_length_mm.value_counts(),y=df.culmen_length_mm.value_counts())

# bivariate


# sns.lineplot(x=df.culmen_length_mm, y=df.body_mass_g)

# multivariate
# sns.pairplot(df)
df.isnull().any()
df.fillna(df.mode())

le = LabelEncoder()
df.sex = le.fit_transform(df.sex)
df.head()

# split
X = df.drop(columns=['culmen_length_mm'], axis=1)
X.head()

y = df.culmen_length_mm
y.head()


scale = MinMaxScaler()
xscale = scale.fit_transform(X)
# X_scaled = pd.DataFrame(xscale, columns=X.columns)
# X_scaled.head()

# train test
# x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.2,random_state=0)
