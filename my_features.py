#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:27:26 2021

@author: bdobkowski
"""

desired_indicators = ['NY.GDP.MKTP.KD.ZG',\
                          'NY.GDP.PCAP.KD',\
                              'NY.GDP.MKTP.KD',\
                                  'NY.GDP.MKTP.KD',\
                                      'SP.POP.TOTL',\
                                          'SP.POP.TOTL']
        
field_names = ['GDP_Growth',\
               'GDP Per Capita',\
               'GDP',\
                       'GDP_Normalized',\
                       'Pop',\
                    'Pop_Normalized']
    
year_begin = 1988
year_end = 2016

validation_year = 2016