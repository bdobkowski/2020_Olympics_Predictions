#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:59:49 2021

@author: bdobkowski
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math

def clean_oly_data(data, wdi, world, indicators, field_names):
    data = data[data.Season == 'Summer']
    data = data.drop(columns=['Height','Weight','Age','Sex','Games'])
    data.Medal.fillna(False)
    
    my_cols = ['Nation','Year'] + \
        field_names + \
            ['Athletes','Athletes_Normalized',\
             'Medals_Last_Games',\
                 'Medals_Last_Games_Normalized',\
                     'Medals','Medals_Normalized']
        
    # Replacing medal types with medal T/F
    data.Medal.replace('Gold',True, inplace=True)
    data.Medal.replace('Silver',True, inplace=True)
    data.Medal.replace('Bronze',True, inplace=True)
    
    
    # Summing medals for each country and creating new DF
    newdf = pd.DataFrame(columns=my_cols)    
    for region in data.region.unique(): # TODO
        # print(region)
        naming_discrepancy = True
        if not type(region)==str:
            if math.isnan(region):
                continue
        for year in data.Year.unique():
            # print(year)
            subset_world = data.loc[(data.Year==year)]
            total_medals_year = np.sum(subset_world.Medal)
            total_athletes_year = len(subset_world['Name'].unique())
                        
            subset = data.loc[(data.Year==year) & (data.region==region)]
            total_medals = np.sum(subset.Medal)
            total_athletes = len(subset['Name'].unique())
            
            total_medals_norm = total_medals / total_medals_year
            total_athletes_norm = total_athletes / total_athletes_year
            
            indicators_list = []
            
            for i in range(len(indicators)):
                ind = indicators[i]
                sub = wdi[wdi['Country Name'] == region]
                sub = sub[sub['Indicator Code'] == ind]
                if len(sub) == 1:
                    my_val = sub[str(year)].values
                    if len(my_val) > 1:
                        raise Exception('more than one data point')                          
                    if 'Normalized' in field_names[i]:
                        subworld  = world[world['Indicator Code'] == ind]
                        world_val = subworld[str(year)].values
                        indicators_list.append(my_val[0] / world_val[0])
                    else:                        
                        indicators_list.append(my_val[0])                                        
                elif len(sub) == 0:
                    indicators_list.append(np.nan)
                else:
                    raise Exception('indicators list greater than 1')
            
                # Figuring out naming discrepancies
                if not np.sum(np.isnan(indicators_list)) == len(indicators_list):
                    naming_discrepancy = False
                    
            newdf = newdf.append(pd.DataFrame([[region, year] + indicators_list + [total_athletes, total_athletes_norm, np.nan, np.nan, total_medals, total_medals_norm]], columns=my_cols))
        
        if naming_discrepancy:
            print(region)
    
    unique_years = np.sort(data.Year.unique())
    
    for i in range(1,len(unique_years)):
        this_year = unique_years[i]
        last_year = unique_years[i-1]
        for nation in newdf[newdf.Year==this_year].Nation.unique():
            df_last = newdf.loc[(newdf.Year==last_year) & (newdf.Nation==nation)]
            medals_last_year = df_last.Medals.values
            medals_last_year_norm = df_last.Medals_Normalized.values
            if not len(medals_last_year) == 1:
                raise Exception('Problem with number last year medals')
            newdf.loc[(newdf.Year==this_year) & (newdf.Nation==nation),'Medals_Last_Games'] = medals_last_year[0]
            newdf.loc[(newdf.Year==this_year) & (newdf.Nation==nation),'Medals_Last_Games_Normalized'] = medals_last_year_norm[0]

    return newdf

def remove_duplicate_medals(data):    
    tempdf = pd.DataFrame(columns=data.columns)
    for year in data.Year.unique():
        print(year)
        for event in data[data.Year==year].Event.unique():
            for medal in ['Gold','Silver','Bronze']:
                subset = data.loc[(data.Year==year) & (data.Event==event) & (data.Medal==medal)]
                if (not subset.empty) and len(subset) > 1 and len(subset.Team.unique()) == 1:
                    tempdf = tempdf.append(subset.iloc[1:])
        
    return_df = data.drop(tempdf.index)
    return_df.to_csv('./RawData/data_no_duplicate.csv')


def correct_names(wdi):
    wdi['Country Name'].replace('United States','USA', inplace=True)
    wdi['Country Name'].replace('Russian Federation','Russia', inplace=True)
    wdi['Country Name'].replace('Iran, Islamic Rep.','Iran', inplace=True)
    wdi['Country Name'].replace('Brunei Darussalam','Brunei', inplace=True)
    wdi['Country Name'].replace('Syrian Arab Republic','Syria', inplace=True)
    wdi['Country Name'].replace('Yemen, Rep.','Yemen', inplace=True)
    wdi['Country Name'].replace('Kyrgyz Republic','Kyrgyzstan', inplace=True)
    wdi['Country Name'].replace('Congo, Dem. Rep.','Democratic Republic of the Congo', inplace=True)
    wdi['Country Name'].replace('Congo, Rep.','Republic of Congo', inplace=True)
    wdi['Country Name'].replace("Cote d'Ivoire",'Ivory Coast', inplace=True)
    wdi['Country Name'].replace("United Kingdom",'UK', inplace=True)
    wdi['Country Name'].replace("St. Kitts and Nevis",'Saint Kitts', inplace=True)
    wdi['Country Name'].replace("Trinidad and Tobago",'Trinidad', inplace=True)
    wdi['Country Name'].replace("St. Vincent and the Grenadines",'Saint Vincent', inplace=True)
    wdi['Country Name'].replace("Egypt, Arab Rep.",'Egypt', inplace=True)
    wdi['Country Name'].replace("Venezuela, RB",'Venezuela', inplace=True)
    wdi['Country Name'].replace("British Virgin Islands",'Virgin Islands, British', inplace=True)
    wdi['Country Name'].replace("Virgin Islands (U.S.)",'Virgin Islands, US', inplace=True)
    wdi['Country Name'].replace("Slovak Republic",'Slovakia', inplace=True)
    wdi['Country Name'].replace("Bahamas, The",'Bahamas', inplace=True)
    wdi['Country Name'].replace("Lao PDR",'Laos', inplace=True)
    wdi['Country Name'].replace("Korea, Dem. People’s Rep.",'North Korea', inplace=True)
    wdi['Country Name'].replace("Korea, Rep.",'South Korea', inplace=True)
    wdi['Country Name'].replace("Cabo Verde",'Cape Verde', inplace=True)
    wdi['Country Name'].replace("Bolivia",'Boliva', inplace=True)
    wdi['Country Name'].replace("Antigua and Barbuda",'Antigua', inplace=True)
    wdi['Country Name'].replace("St. Lucia",'Saint Lucia', inplace=True)
    wdi['Country Name'].replace("Micronesia, Fed. Sts.",'Micronesia', inplace=True)
    wdi['Country Name'].replace("North Macedonia",'Macedonia', inplace=True)
    wdi['Country Name'].replace("Gambia, The",'Gambia', inplace=True)
    wdi['Country Name'].replace("Eswatini",'Swaziland', inplace=True)
    
    return wdi

def main(year_begin, year_end, desired_indicators, field_names):
    data    = pd.read_csv('./RawData/data_no_duplicate.csv')
    regions = pd.read_csv('./RawData/noc_regions.csv')
    wdi     = pd.read_csv('./RawData/WDIData.csv')
    
    # This variable is included to calculate medals in prev games
    year_begin = year_begin - 4
    
    data = data[data.Year >= year_begin]
    data = data[data.Year <= year_end]
    
    merged = pd.merge(data, regions, on='NOC', how='left')
    
    # Naming Discrepancies
    wdi = correct_names(wdi)
    
    world = wdi[wdi['Country Name'] == 'World']
    wdi = wdi[wdi['Country Name'].isin(regions.region.unique())]
    
    wdi = wdi[wdi['Indicator Code'].isin(desired_indicators)]
    world = world[world['Indicator Code'].isin(desired_indicators)]
    
    years_list = [str(i) for i in merged.Year.unique() if (i>=year_begin and i<=year_end)]
    temp = wdi[years_list].notna()
    wdi = wdi[temp.eq(1).all(axis=1)]
    
    cleandf = clean_oly_data(merged, wdi, world, desired_indicators, field_names)
    
    # Getting rid of examples with NaN values
    sns.heatmap(cleandf.isnull())
    cleandf = cleandf.dropna(axis=0)
    sns.heatmap(cleandf.isnull())
    
    # usa = cleandf.loc[cleandf.Nation=='USA']
    # italy = cleandf.loc[(cleandf.Nation=='Italy')]
    # china = cleandf.loc[(cleandf.Nation=='China')]
    # print(np.sum(usa.Medals))
    # print(np.sum(italy.Medals))
    
    cleandf.to_csv('./RawData/cleaned_data.csv')
    return cleandf

def train_test_split(data, validation_year, normalized=False):
    training_data = data[data.Year!=validation_year]
    valid_data = data[data.Year==validation_year]
    
    if normalized:
        y = 'Medals_Normalized'
    else:
        y = 'Medals'
        
    cols_to_drop = ['Unnamed: 0','Nation','Year','Medals','Medals_Normalized']
    
    x_train = training_data.drop(columns=cols_to_drop)
    y_train = training_data[y]
    
    x_valid = valid_data.drop(columns=cols_to_drop)
    y_valid = valid_data[y]
    
    return x_train, y_train, x_valid, y_valid

if __name__ == '__main__':
    desired_indicators = ['NY.GDP.MKTP.KD.ZG',\
                          'NY.GDP.PCAP.KD',\
                          'SP.POP.TOTL',\
                              'SP.POP.TOTL']
        
    field_names = ['GDP 1',\
                   'GDP 2',\
                       'Pop',\
                    'Pop_Normalized']
        
    year_begin = 1988
    year_end   = 2016
    data = main(year_begin, year_end, desired_indicators, field_names)
    
