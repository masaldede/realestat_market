# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:26:57 2024

@author: bahadir sahin
"""

############ SGD Real Estate Market Prediction ###########

# Let's import the necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Downloading the dataset:
df = pd.read_excel("https://www.dropbox.com/s/luoopt5biecb04g/SATILIK_EVI.xlsx?dl=1")
df.head()
print("dataset downloaded")

# Let's define the target (y) and feature variables (X):
X = df[['Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı']]
y = df['Fiyat']
print("target and features defined")

# SGD is sensitive to the scale of feature and target variables. Therefore,
# we apply standard scaling to the values of all variables:
from sklearn.preprocessing import MinMaxScaler

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = X_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()  # Corrected this line
print("standart scaling applied")

# Splitting the dataset into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("training and test sets splitting")

# Download the necessary library for SGD Regression and define the prediction model:
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(random_state=42, penalty='elasticnet', max_iter=150000, tol=1e-6)  # Further increased max_iter and set tol

# Grid Search for SGD Regression:
from sklearn.model_selection import GridSearchCV
print("grid seach completed")

parametreler = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'eta0': [0.0001, 0.001, 0.01, 0.1],
    'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
}

sgd_reg_GS = GridSearchCV(estimator=sgd_reg, param_grid=parametreler, n_jobs=-1, scoring='r2', cv=5)
sgd_reg_GS.fit(X_train, y_train.ravel())  # Corrected this line

# Results of Hyperparameter Optimization (Grid Search):
print("\nGrid search results\n")
print(sgd_reg_GS.best_params_)
print(sgd_reg_GS.best_estimator_)
print(sgd_reg_GS.best_score_)

#cross validation
from sklearn.model_selection import cross_val_score
skorlar = cross_val_score(estimator=sgd_reg_GS.best_estimator_, X=X_train, y=y_train, cv=5)  # Use cv variable here

#cross-validation results 
print("\nCross-validation scores\n")
print(skorlar)
print(f"Mean CV Score: {skorlar.mean()}")
print(f"Standard Deviation of CV Scores: {skorlar.std()}")

#visualition of cross-validation results
from yellowbrick.model_selection import CVScores
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)  # Set shuffle=True
model = sgd_reg_GS.best_estimator_
visualizer = CVScores(model, cv=cv, scoring="r2")
visualizer.fit(X_train, y_train)
visualizer.show()
print("cross-validation result visualiated")

#check sgd reg. parameters
sgd_reg_GS.best_estimator_.intercept_ #constant term
sgd_reg_GS.best_estimator_.coef_ #prediction parameter coeff 

print(sgd_reg_GS.best_estimator_.intercept_)
print(sgd_reg_GS.best_estimator_.coef_)

# Let's look at the R2 scores for the training and test sets:
sgd_reg_train_score = sgd_reg_GS.best_estimator_.score(X_train, y_train)
sgd_reg_test_score = sgd_reg_GS.best_estimator_.score(X_test, y_test)
print("\nr2 scores\n")
print(f"R2 for training set: {sgd_reg_train_score}")
print(f"R2 for test set: {sgd_reg_test_score}\n")

# Let's predict the prices for the entire dataset with the prediction model:
df['Predicted_Price'] = sgd_reg_GS.best_estimator_.predict(X)

# Reverse the scaling of the target and predicted prices to convert to original values:
df['Predicted_Price'] = y_scaler.inverse_transform(df['Predicted_Price'].values.reshape(-1, 1))

print(df[['Fiyat', 'Predicted_Price']].head(5))  # Corrected this line

#Graph: Comparison of actual and predicted prices of the houses
plt.figure(figsize=(10, 6))
plt.style.use('fivethirtyeight')
plt.title('SGD (Stochastic Gradient Descent) Regression')
plt.plot(df['Fiyat'], df.index.values, label='Price')
plt.plot(df['Predicted_Price'], df.index.values, 'ro', label='Predicted_Price')
plt.xlabel('For Sale Houses', fontsize=15)
plt.ylabel('Predicted/Actual Prices', fontsize=15)
plt.legend(fontsize=13, loc='lower right')
plt.show()

# Comparison of SGD Optimization and Linear Regression Performance:
MSE_train = mean_squared_error(y_train, sgd_reg_GS.best_estimator_.predict(X_train))
MSE_test = mean_squared_error(y_test, sgd_reg_GS.best_estimator_.predict(X_test))

MSE_train_original = y_scaler.inverse_transform(MSE_train.reshape(-1, 1))
MSE_test_original = y_scaler.inverse_transform(MSE_test.reshape(-1, 1))

print(f"MSE Train: {MSE_train_original}")
print(f"MSE Test: {MSE_test_original}")

# Linear Regression:
from sklearn.linear_model import LinearRegression

X = df[['Oda_Sayısı', 'Net_m2', 'Katı', 'Yaşı']]
y = df['Fiyat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR = LinearRegression()
LR.fit(X_train, y_train)

y_train_pred_LR = LR.predict(X_train)
y_test_pred_LR = LR.predict(X_test)

mse_train_LR = mean_squared_error(y_train, y_train_pred_LR)
mse_test_LR = mean_squared_error(y_test, y_test_pred_LR)

r2_train_LR = LR.score(X_train, y_train)
r2_test_LR = LR.score(X_test, y_test)

print(f"LR Train MSE: {mse_train_LR}")
print(f"LR Test MSE: {mse_test_LR}")
print(f"LR Train R2: {r2_train_LR}")
print(f"LR Test R2: {r2_test_LR}")