import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# k = size of each validation set
def cross_validate(input_data, k):
    score = 0.0
    for i in range(int(input_data.shape[0] / k)):
        # print("i = " + str(i))
        # print("low: " + str(i * k))
        # print("hi: " + str((i + 1) * k - 1))
        validation_set = input_data.iloc[range(i * k, (i + 1) * k)]
        training_set = input_data.drop(input_data.index[range(i * k, (i + 1) * k)])
        # print(training_set)
        x_train1 = training_set[rel_cols[:-1]].iloc[:, 0:].values
        y_train1 = training_set.iloc[:, -1].values
        # print(y_train1)
        # print(y_train1.shape[0])
        x_test1 = validation_set[rel_cols[:-1]].iloc[:, 0:].values
        y_test1 = validation_set.iloc[:, -1].values
        score = score + pred(x_train1, y_train1, x_test1, y_test1)
    return score / (input_data.shape[0] / k)


def pred(xtrain, ytrain, xtest, ytest):
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=300, random_state=0)
    regressor.fit(xtrain, ytrain)
    this_score = regressor.score(xtrain, ytrain)
    print("The score is: " + str(regressor.score(xtrain, ytrain)))
    return this_score


try:
    pdTest = pd.read_csv('test.csv')
except:
    print("test.csv load failed")

try:
    pdTrain = pd.read_csv('train.csv')
except:
    print("train.csv load failed")

try:
    pd_data = pd.read_csv('data.csv')
except:
    print("data.csv load failed")

data = pd_data.head(40).filter(
    items=['abb', 'bby', 'van', 'sur', 'rmd', 'bedrooms', 'bathrooms', 'lotSize', 'builtYear', 'actualValue'])
# print(data.shape[0])
train = pdTrain.head(30).filter(
    items=['abb', 'bby', 'van', 'sur', 'rmd', 'bedrooms', 'bathrooms', 'lotSize', 'builtYear', 'actualValue'])
test = pdTest.head(10).filter(
    items=['abb', 'bby', 'van', 'sur', 'rmd', 'bedrooms', 'bathrooms', 'lotSize', 'builtYear', 'actualValue'])

# trainAreas = pdTrain.filter('area').values

# print(train)

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
# score = regressor.score(X_train, y_train)
# print("The score for 30/10: " + str(regressor.score(X_train, y_train)))


n = 5  # <- the 5 here is the number of iterations we want
k = int(data.shape[0] / n)

print("The score for n = " + str(n) + " is " + str(cross_validate(data, k)))


# Plot y_test vs y_pred
plt.figure(figsize=(12,8))
plt.plot(y_test, color='red')
plt.plot(y_pred, color='blue')
plt.show()

