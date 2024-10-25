import numpy as np
from sklearn.model_selection import cross_val_score
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

model = simple_regression(df[['Estimate (Land)']], df['Sale Price'])

print("Simple Regression Error: ")
print(cross_validate(model, df[['Estimate (Land)']], df['Sale Price']))