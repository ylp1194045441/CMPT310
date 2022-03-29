import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    pdTest = pd.read_csv('test.csv')
except:
    print("test.csv load failed")

try:
    pdTrain = pd.read_csv('train.csv')
except:
    print("train.csv load failed")

test = pdTest.head(10).filter(items=['bedrooms', 'bathrooms', 'lotSize', 'builtYear', 'actualValue'])
testAreas = pdTest.head(10).filter(items='area').values
train = pdTrain.head(10).filter(items=['bedrooms', 'bathrooms', 'lotSize', 'builtYear', 'actualValue'])
trainAreas = pdTrain.filter('area').values

print("csv read success")

corr = train.corr()
# print(corr.actualValue)
rel_vars = corr.actualValue[corr.actualValue > -1]
rel_cols = list(rel_vars.index.values)

corr2 = train[rel_cols].corr()
plt.figure(figsize=(8, 8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size': 10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
# plt.show()

X_train = train[rel_cols[:-1]].iloc[:, 0:].values
y_train = train.iloc[:, -1].values

X_test = test[rel_cols[:-1]].iloc[:, 0:].values
y_test = test.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, y_train)

print(regressor.score(X_train, y_train))

y_pred = regressor.predict(X_test)
# Plot y_test vs y_pred
plt.figure(figsize=(12,8))
plt.plot(y_test, color='red')
plt.plot(y_pred, color='blue')
plt.show()