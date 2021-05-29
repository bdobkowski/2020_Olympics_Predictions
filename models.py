#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 17:03:39 2021

@author: bdobkowski
"""
import pandas as pd
import numpy as np
import clean_data
from my_features import year_begin, year_end, desired_indicators, field_names
from matplotlib import pyplot as plt
import utils
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor as NN
import statsmodels.api as sm
from patsy import dmatrices
from patsy import dmatrix

linear_model = LR(fit_intercept=True)
linear_model.fit(xt, yt)
y_predict = linear_model.predict(xv)

utils.plot(y_valid, y_predict, 'Linear Model Predictions',
           line=True,save_path='./Plots/linearModPred.ps')

i = 0

for index in y_valid.index:    
    print(data.iloc[index].Nation + '\t' + str(y_valid[index]) + '\t' + str(y_predict[i]))
    i += 1
    
print('\nAccuracy: ' + str(sq_loss(y_valid, y_predict)))
    
print('===========================\n')
    
for i in range(len(x_train.columns)):    
    print(str(x_train.columns[i]) + ': ' + str(linear_model.coef_[i]))
    i += 1
    
print('===========================\n')

y_train_p, x_train_p = dmatrices('Medals ~ GDP_Per_Capita + Area + GDP_Growth + Pop + Medals_Last_Games_Normalized + GDP + Athletes + Pop_Normalized + GDP_Normalized', data, return_type='dataframe')
x_train_d = dmatrix(x_train, return_type='dataframe')
y_train_d = dmatrix(y_train, return_type='dataframe')
x_valid_d = dmatrix(x_valid, return_type='dataframe')
y_valid_d = dmatrix(y_valid, return_type='dataframe')

zip_training_results = sm.ZeroInflatedPoisson(endog=y_train_d, exog=x_train_d, exog_infl=x_train_d, inflation='logit').fit(method='bfgs',maxiter=1000)
# neg_res = sm.ZeroInflatedNegativeBinomialP(endog=y_train_d, exog=x_train_d, exog_infl=x_train_d, inflation='logit', missing='drop').fit(method='bfgs',maxiter=1000)
poisson_results = sm.GLM(endog=y_train_d, exog=x_train_d, family=sm.families.Poisson()).fit()
# print(poisson_results.summary())
# print(zip_training_results.summary())

y_predict_zip = zip_training_results.predict(x_valid_d, exog_infl=x_valid_d)
y_predict_poisson = poisson_results.predict(x_valid_d)
# y_predict_neg = neg_res.predict(x_valid_d, exog_infl=x_valid_d)

print('\nZIP Accuracy: ' + str(sq_loss(y_valid, y_predict_zip)))
print('===========================\n')
print('\nPoisson Accuracy: ' + str(sq_loss(y_valid, y_predict_poisson)))
print('===========================\n')
# print('\Zero Neg Binomial Accuracy: ' + str(sq_loss(y_valid, y_predict_neg)))
# print('===========================\n')

yt_bool = np.zeros(len(yt))
yt_bool[yt!=0] = 1
yv_bool = np.zeros(len(yv))
yv_bool[yv!=0] = 1

# for col in range(xt.shape[1]):
#     print(x_train.columns[col])
#     print(np.max(xt[:,col]))
#     xt_ = np.delete(xt, col, axis=1)
#     # logit_res = LogisticRegression(max_iter=10000).fit(xt_,yt_bool)
#     zip_training_results = sm.ZeroInflatedPoisson(endog=yt, exog=xt_, exog_infl=xt_, inflation='logit').fit(method='bfgs')

    
logit_res = LogisticRegression(max_iter=10000).fit(xt,yt_bool)
logit_res.predict(xv)

# zip_training_results = sm.ZeroInflatedPoisson(endog=y_train, exog=x_train, exog_infl=x_train, inflation='logit').fit()

# print(zip_training_results.summary())
# ridge_model = Ridge(fit_intercept=True)
# ridge_model.fit(xt, yt)
# y_predict = ridge_model.predict(xv)

# utils.plot(y_valid, y_predict, 'Ridge Model Predictions', line=True)

# lasso_model = Lasso(fit_intercept=True)
# lasso_model.fit(xt, yt)
# y_predict = lasso_model.predict(xv)

# utils.plot(y_valid, y_predict, 'Lasso Model Predictions', line=True)

# i = 0
# for index in y_valid.index:    
#     print(data.iloc[index].Nation + '\t' + str(y_valid[index]) + '\t' + str(y_predict[i]))
#     i += 1
