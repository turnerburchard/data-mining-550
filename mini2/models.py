import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from scipy.interpolate import CubicSpline
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
    selector = SelectKBest(score_func=f_regression, k=k)
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
    print("Fitting polynomial regression model with degree:", degree)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model


X = df.drop(columns=['Sale Price'])
y = df['Sale Price']
degree = 1  # Might need tuning
poly_model = polynomial_regression(X, y, degree)

print("Polynomial Regression Error: ")
print(cross_validate(poly_model, X, y))


# TODO doesn't work
# Scipy cubic spline interpolation
class CubicSplineRegressor:
    def __init__(self):
        self.models = []

    def fit(self, X, y):
        self.models = []
        for column in X.columns:
            X_sorted, y_sorted = zip(*sorted(zip(X[column], y)))
            X_sorted, y_sorted = np.unique(X_sorted, return_index=True)
            y_sorted = np.array(y)[y_sorted]

            model = CubicSpline(X_sorted, y_sorted, bc_type='natural')  
            self.models.append(model)
        return self

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i, column in enumerate(X.columns):
            spline_model = self.models[i]
            y_pred += spline_model(X[column])
        return y_pred
    
    
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    

X = df.drop(columns=['Sale Price'])
y = df['Sale Price']
spline_model = CubicSplineRegressor()

print("Spline Regression Error: ")
print(cross_validate(spline_model, X, y))


# Regularization
def lassocv_regularization(X, y):
    lasso_cv = LassoCV(cv=5).fit(X, y)
    return lasso_cv

# Lasso regularization
def lasso_regularization(X, y, alpha=1):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model


X = df.drop(columns=['Sale Price'])
y = df['Sale Price']
#lasso_model = lasso_regularization(X, y)
lasscv_model = lassocv_regularization(X, y)

print("Lasso Regression Error: ")
#print(cross_validate(lasso_model, X, y))
print(lasscv_model.score(X, y)) # this uses the R^2 score

def ridgecv_regularization(X, y):
    ridgecv_model = RidgeCV(cv=5).fit(X, y)
    return ridgecv_model

# Ridge regularization
def ridge_regularization(X, y, alpha=1):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df.drop(columns=['Sale Price']))
y = df['Sale Price']
#ridge_model = ridge_regularization(X, y)
ridgecv_model = ridgecv_regularization(X_scaled, y)

print("Ridge Regression Error: ")
#print(cross_validate(ridge_model, X, y))
print(ridgecv_model.score(X_scaled, y)) # this is the R^2 score
