#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:27:26 2021

@author: bdobkowski
"""

# desired_indicators = \
# ['NY.GDP.MKTP.KD.ZG',\
# 'NY.GDP.PCAP.KD',\
# 'NY.GDP.MKTP.KD',\
# 'NY.GDP.MKTP.KD',\
# 'SP.POP.TOTL',\
# 'SP.POP.TOTL']
    
# field_names = \
# ['GDP_Growth',\
# 'GDP Per Capita',\
# 'GDP',\
# 'GDP_Normalized',\
# 'Pop',\
# 'Pop_Normalized']
    
desired_indicators = \
['NY.GDP.MKTP.KD.ZG',\
'NY.GDP.PCAP.KD',\
'NY.GDP.MKTP.KD',\
'NY.GDP.MKTP.KD',\
'SP.POP.TOTL',\
'SP.POP.TOTL',\
'AG.SRF.TOTL.K2',\
'SP.POP.GROW']
        
field_names = \
['GDP_Growth',\
'GDP_Per_Capita',\
'GDP',\
'GDP_Normalized',\
'Pop',\
'Pop_Normalized',\
'Area',\
'Pop_Growth']
    
year_begin = 1988
year_end = 2016

validation_year = 2016