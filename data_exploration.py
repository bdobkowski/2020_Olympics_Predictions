#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:54:33 2021

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
from sklearn.neural_network import MLPRegressor as NN


sq_loss = lambda true_data, predict_data : np.sum((true_data - predict_data)**2)

clean                   = False
remove_duplicate_medals = False
raw_data                = pd.read_csv('./RawData/athlete_events.csv')

# Removing multiple medals for each team sport
if remove_duplicate_medals:
    clean_data.remove_duplicate_medals(raw_data)

if not clean:
    data = pd.read_csv('./RawData/cleaned_data.csv')
else:
    data = clean_data.main(year_begin, year_end, desired_indicators, field_names)

x_train, y_train, x_valid, y_valid = clean_data.train_test_split(data, year_end)

xt = x_train.to_numpy()
yt = y_train.to_numpy()
xv = x_valid.to_numpy()
yv = y_valid.to_numpy()

# for feature in data.drop(columns=['Nation','Year','Medals','Medals_Normalized']).columns:
#     utils.plot(data[feature], data['Medals'])
    
for nation in data.Nation.unique():
    x = data[data.Nation==nation].Year
    y = data[data.Nation==nation].Medals
    if np.sum(y) > 400:
        utils.plot(x, y, nation)
        
linear_model = LR(fit_intercept=True)
linear_model.fit(xt, yt)
y_predict = linear_model.predict(xv)

utils.plot(y_valid, y_predict, 'Linear Model Predictions')

i = 0

for index in y_valid.index:    
    print(data.iloc[index].Nation + '\t' + str(y_valid[index]) + '\t' + str(y_predict[i]))
    i += 1
    
    
print(sq_loss(y_valid, y_predict))

# ridge_model = Ridge(fit_intercept=True)
# ridge_model.fit(xt, yt)
# y_predict = ridge_model.predict(xv)

# utils.plot(y_valid, y_predict, 'Linear Model Predictions')

# i = 0
# for index in y_valid.index:    
#     print(data.iloc[index].Nation + '\t' + str(y_valid[index]) + '\t' + str(y_predict[i]))
#     i += 1
    
# print(sq_loss(y_valid, y_predict))