# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:41:49 2018

@author: KoalaChelsea 
"""

## Importing pandas and calling it pd
## Importing numpy and calling it np
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
#%%
## Read this data into Python using pandas as a dataframe.
resturantdf = pd.read_csv("RawData/Resturant_Full_Infomation.csv")
#print(resturantdf.head())
#print(resturantdf.columns)
## Frist new feature:
#print(resturantdf['review_count'].describe())
#plt.hist(resturantdf['review_count'])
## Polularity of the resturant using review count
def first_feature(resturantdf):
    resturantdf['popularity']='NaN'
    for i in resturantdf.id.index:
        if resturantdf.loc[i,'review_count'] < 10:
            resturantdf.loc[i,'popularity'] = 'biased'
        elif 10 <= resturantdf.loc[i,'review_count'] < 200:
            resturantdf.loc[i,'popularity'] = 'F'
        elif 200 <= resturantdf.loc[i,'review_count'] < 400:
            resturantdf.loc[i,'popularity'] = 'E'
        elif 400 <= resturantdf.loc[i,'review_count'] < 600:
            resturantdf.loc[i,'popularity'] = 'D'
        elif 600 <= resturantdf.loc[i,'review_count'] < 800:
            resturantdf.loc[i,'popularity'] = 'C'
        elif 800 <= resturantdf.loc[i,'review_count'] < 1000:
            resturantdf.loc[i,'popularity'] = 'B'
        elif resturantdf.loc[i,'review_count'] >= 1000:
            resturantdf.loc[i,'popularity'] = 'A'
        else:
            print("test for invalid")
    return resturantdf

def second_feature(row):
    price=row['price']
    if (len(price)==1):
        return 1
    elif (len(price)==2):
        return 2
    elif(len(price)==3):
        return 3
    elif(len(price)==4):
        return 4
    
def ifnull(var, val):
  if math.isnan(var):
    return val
  return var
   
def third_feature(row):
    transport_spot=ifnull(row['bus_station'],0)+ifnull(row['subway_station'],0)+ \
    ifnull(row['train_station'],0)+ifnull(row['taxi_stand'],0)
    if(transport_spot==0):
        return 'Very_Uncovenient'
    elif(transport_spot>0 and transport_spot<=10):
        return 'Somewhat_Unconvenient'
    elif (transport_spot>10 and transport_spot<=30):
        return 'Somewhat_convenient'
    elif (transport_spot>30 and transport_spot<=60):
        return 'Convenient'
    else:
        return 'Super_Convenient'
#%%
first_feature(resturantdf)
print("New Feature 1 and counts:")
print(resturantdf['popularity'].value_counts(dropna = False))
resturantdf['price_category']=resturantdf.apply (lambda row: second_feature(row),axis=1)
print("New Feature 2 and counts:")
print(resturantdf['price_category'].value_counts(dropna = False))
print("New Feature 3 and counts:")
resturantdf['transportation']=resturantdf.apply (lambda row: third_feature(row),axis=1)
print(resturantdf['transportation'].value_counts(dropna = False))










