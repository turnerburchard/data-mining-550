import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
import preprocessing

df = preprocessing.preprocess()


# 5-fold Cross Validation
def cross_validate(model, X, y, n_folds=5):
    scores = cross_val_score(model, X, y, cv=n_folds, scoring='neg_mean_squared_error')
    return int(np.sqrt((-np.mean(scores))))


# Simple Regression
from sklearn.linear_model import LinearRegression

def simple_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

simple_model = simple_regression(df[['Estimate (Land)']], df['Sale Price'])

print("Simple Regression Error: ")
print(cross_validate(simple_model, df[['Estimate (Land)']], df['Sale Price']))


# Multiple Regression with all features

def multiple_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

multiple_model = multiple_regression(df.drop(columns=['Sale Price']), df['Sale Price'])

print("Multiple Regression Error: ")
print(cross_validate(multiple_model, df.drop(columns=['Sale Price']), df['Sale Price']))


def subset_selection(X, y, k):
    selector = SelectKBest(k=k)
    return selector.fit_transform(X, y)


# Multiple Regression with subset of features
k = 15
X = subset_selection(df.drop(columns=['Sale Price']), df['Sale Price'], k)
subset_model = multiple_regression(X, df['Sale Price'])

print("Subset Regression Error: ")
print(cross_validate(subset_model, X, df['Sale Price']))
