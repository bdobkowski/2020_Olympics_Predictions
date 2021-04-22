#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:54:33 2021

@author: bdobkowski
"""
import pandas as pd
import numpy as np
import CleanData
from MyFeatures import year_begin, year_end, desired_indicators, field_names
from matplotlib import pyplot as plt
import Utils

clean_data = True

if not clean_data:
    data = pd.read_csv('./RawData/cleaned_data.csv')
else:
    data = CleanData.main(year_begin, year_end, desired_indicators, field_names)
    
for feature in data.drop(columns=['Nation','Year','Medals','Medals_Normalized']).columns:
    Utils.plot(data[feature], data['Medals'])