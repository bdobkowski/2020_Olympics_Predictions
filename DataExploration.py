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


def clean_oly_data(data, wdi, indicators, field_names):
    data = data.drop(columns=['Height','Weight','Age','Sex','Name','Games'])
    data = data[data.Season == 'Summer']
    data.Medal.fillna(False)
    
    my_cols = ['Nation','Year'] + field_names + ['Medals']
    
    newdf = pd.DataFrame(columns=my_cols)
    tempdf = pd.DataFrame(columns=data.columns)
    
    # Removing multiple medals for each team sport
    for year in data.Year.unique(): # TODO
        print(year)
        for event in data[data.Year==year].Event.unique():
            for medal in ['Gold','Silver','Bronze']:
                subset = data.loc[(data.Year==year) & (data.Event==event) & (data.Medal==medal)]
                if (not subset.empty) and len(subset) > 1 and len(subset.Team.unique()) == 1:
                    tempdf = tempdf.append(subset.iloc[1:])
                    # print(subset)
        
    data = data.drop(tempdf.index)
    
    data.Medal.replace('Gold',True, inplace=True)
    data.Medal.replace('Silver',True, inplace=True)
    data.Medal.replace('Bronze',True, inplace=True)
        
    # Summing medals for each country and creating new DF
    for region in data.region.unique(): # TODO
        # print(region)
        naming_discrepancy = True
        for year in data.Year.unique():
            # print(year)
            subset = data.loc[(data.Year==year) & (data.region==region)]
            # idx_year = [idx for idx, el in enumerate(data.Year) if el==year]
            # idx_noc  = [idx for idx, el in enumerate(data.NOC) if el==noc]
            # idx = np.intersect1d(idx_year, idx_noc)
            total_medals = np.sum(subset.Medal)
            indicators_list = []
            
            for ind in indicators:
                sub = wdi[wdi['Country Name'] == region]
                sub = sub[sub['Indicator Code'] == ind]
                if len(sub) == 1:
                    my_val = sub[str(year)].values
                    if len(my_val) > 1:
                        raise Exception('more than one data point')                    
                    indicators_list.append(my_val[0])
                elif len(sub) == 0:
                    indicators_list.append(np.nan)
                else:
                    raise Exception('indicators list greater than 1')
            # print(total_medals)
            # print(indicators_list)
            
            # Figuring out naming discrepancies
                if not np.sum(np.isnan(indicators_list)) == len(indicators_list):
                    naming_discrepancy = False
            newdf = newdf.append(pd.DataFrame([[region, year] + indicators_list + [total_medals]], columns=my_cols))
        if naming_discrepancy:
            print(region)
    return newdf

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

data = pd.read_csv('./RawData/athlete_events.csv')
regions = pd.read_csv('./RawData/noc_regions.csv')
data = data[data.Year >= 1976]
data = data[data.Year <= 2016]

merged = pd.merge(data, regions, on='NOC', how='left')

desired_indicators = ['NY.GDP.MKTP.KD.ZG',\
                      'NY.GDP.PCAP.KD',\
                      'SP.POP.TOTL']
    
field_names = ['GDP 1',\
               'GDP 2',\
                   'Pop']

# desired_indicators = ['NY.GDP.MKTP.KD.ZG']

wdi = pd.read_csv('./RawData/WDIData.csv')

# Naming Discrepancies
wdi = correct_names(wdi)


world = wdi[wdi['Country Name'] == 'World']
wdi = wdi[wdi['Country Name'].isin(regions.region.unique())]

wdi = wdi[wdi['Indicator Code'].isin(desired_indicators)]
world = world[world['Indicator Code'].isin(desired_indicators)]

years_list = [str(i) for i in merged.Year.unique() if (i>=1976 and i<=2016)]
temp = wdi[years_list].notna()
wdi = wdi[temp.eq(1).all(axis=1)]

cleandf = clean_oly_data(merged, wdi, desired_indicators, field_names)

# Getting rid of examples with NaN values
sns.heatmap(cleandf.isnull())
cleandf = cleandf.dropna(axis=0)
sns.heatmap(cleandf.isnull())

usa = cleandf.loc[cleandf.Nation=='USA']
italy = cleandf.loc[(cleandf.Nation=='Italy')]
china = cleandf.loc[(cleandf.Nation=='China')]
print(np.sum(usa.Medals))
print(np.sum(italy.Medals))

# sns.heatmap(data.isnull())
# sns.heatmap(cleandf.Medals)