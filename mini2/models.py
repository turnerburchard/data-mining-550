import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from scipy.interpolate import interpolate
import preprocessing

df = preprocessing.preprocess()


# TODO List
# Hyperparamter Tuning for [subset selection k, PCA k, polynomial degree, lasso, ridge] and more?


# 5-fold Cross Validation
def cross_validate(model, X, y, n_folds=5):
    scores = cross_val_score(model, X, y, cv=n_folds, scoring='neg_mean_squared_error')
    # return RMSE of sale price
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


def pca(X, y, k):
    model = PCA(n_components=k)
    model = model.fit_transform(X, y)

    return model

k = 15
X = pca(df.drop(columns=['Sale Price']), df['Sale Price'], k)
pca_model = multiple_regression(X, df['Sale Price'])

print("PCA Regression Error: ")
print(cross_validate(pca_model, X, df['Sale Price']))


# Non-linear Models - Polynomial and Smooth spline models

# Polynomial regression
def polynomial_regression(X, y, degree):
    print("Fitting polynomial regression model with degree: ", degree)
    model = PolynomialFeatures(degree=degree)
    model.fit_transform(X)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(model, y)

    return poly_reg_model


X = df.drop(columns=['Sale Price'])
y = df['Sale Price']
degree = 5
poly_model = polynomial_regression(X, y, degree)

print("Polynomial Regression Error: ")
print(cross_validate(poly_model, X, y))


# Scipy cubic spline interpolation
def smooth_spline_regression(X, y):
    model = interpolate.splrep(X, y)

    return model


X = df.drop(columns=['Sale Price'])
y = df['Sale Price']
spline_model = smooth_spline_regression(X, y)

print("Spline Regression Error: ")
print(cross_validate(spline_model, X, y))


# Regularization

# Lasso regularization
def lasso_regularization(X, y, alpha=1):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model


X = df.drop(columns=['Sale Price'])
y = df['Sale Price']
lasso_model = lasso_regularization(X, y)

print("Lasso Regression Error: ")
print(cross_validate(lasso_model, X, y))


# Ridge regularization
def ridge_regularization(X, y, alpha=1):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


X = df.drop(columns=['Sale Price'])
y = df['Sale Price']
ridge_model = ridge_regularization(X, y)

print("Ridge Regression Error: ")
print(cross_validate(ridge_model, X, y))

