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


sq_loss = lambda true_data, predict_data : 1/len(true_data) * np.sum((true_data - predict_data)**2)

clean                   = False
remove_duplicate_medals = False
save_normalized         = True
raw_data                = pd.read_csv('./RawData/athlete_events.csv')

# Removing multiple medals for each team sport
if remove_duplicate_medals:
    clean_data.remove_duplicate_medals(raw_data)

if not clean:
    data = pd.read_csv('./RawData/cleaned_data.csv')
else:
    data = clean_data.main(year_begin, year_end, desired_indicators, field_names)
    
# Resizing Units
data.GDP = data.GDP/1000000000000
data.Pop = data.Pop/1000000000
data.Area = data.Area/1000000
data.GDP_Per_Capita = data.GDP_Per_Capita/1000
# data.Athletes[data.Athletes!=0] = np.log(data.Athletes[data.Athletes!=0])
# data.Athletes_Normalized[data.Athletes_Normalized!=0] = np.log(data.Athletes_Normalized[data.Athletes_Normalized!=0])

# TODO figure this out
if 'Unnamed: 0' in data.columns: data = data.drop(columns=['Unnamed: 0'])

if save_normalized:
    data.to_csv('./RawData/cleaned_data_units.csv', index=False)

x_train, y_train, x_valid, y_valid = clean_data.train_test_split(data, year_end)

xt = x_train.to_numpy()
yt = y_train.to_numpy()
xv = x_valid.to_numpy()
yv = y_valid.to_numpy()

utils.plot_hist(data, 'Medals', save_path='./Plots/hist.ps')

for feature in data.drop(columns=['Nation','Medals']).columns:
    utils.plot(data[feature], data['Medals'],
               save_path='./Plots/'+feature+'.ps')
    
for nation in data.Nation.unique():
    x = data[data.Nation==nation].Year
    y = data[data.Nation==nation].Medals
    if np.sum(y) > 400:
        utils.plot(x, y, nation, save_path='./Plots/'+nation+'.pdf')
        
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