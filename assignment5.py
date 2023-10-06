import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Mall_Customers.csv')
df.info()
print(df.isnull().sum())
# there is no  null values


x = df.iloc[:, [3, 4]]


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y = kmeans.fit_predict(x)

print(y)
q1, q3 = np.percentile(df['Annual Income (k$)'], [25, 75])
Iqr = q3-q1
L_bound = q1 - (1.5 * Iqr)
U_bound = q3 - (1.5 * Iqr)

U_value = df['Annual Income (k$)'].median()
df['Annual Income (k$)'] = np.where(df['Annual Income (k$)']
                                    > U_bound, U_value, df['Annual Income (k$)'])
sns.boxplot(df['Annual Income (k$)'])

# correlation
corr = df.corr()

plt.figure(dpi=130)
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.show()
# train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
