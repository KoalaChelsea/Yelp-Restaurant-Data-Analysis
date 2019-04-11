#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:17:08 2018

@author: Shaoyu
"""

#%%
# =============================================================================
# This script is to access th cleaness of the dataset by looking at the 
# completeness, meaningfulness of each column. 
# We will analyze the data after join. 
# For now we will assume equal weitage of each feature columns. 
# =============================================================================

# =============================================================================
# Methods in Calculating the overall cleaniness index for the dataset: 
# There are 59 Columns in the datasets, each column will carry 10 points.
# Based on the heuristics and understanding of the data, we have predefined 
# a list indidate whether it is a  context column and the max min values.
# If it is not null and out of range, we say that the particular cell is out wrong.
# The base score will be (1-missing rate) * 10. 
# For columns with wrong values, each wrong entry will contribute a 0.5 point 
# deduction untill the base point becomes 0. In other words, we will think certain 
# column is very bad if it has more than 20 wrong values. 
# A total score will then be sum up together, and divided by the tota
# =============================================================================

#%%
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#%% 

def DataCleanessCheck(my_data):
    print("The Data Schema will be: ")
    print(my_data.dtypes)
    print("\n\n")
    print("The Summary metrics for numeric columns will be:")
    print(my_data.describe())
    print("\n\n")
    print("The Data Missing Rate will be:")
    missing_rate=my_data.isnull().sum()/my_data.shape[0]
    print(missing_rate)
    print("\n\n")
    missing_rate=missing_rate.to_dict()
    #To load in the pre-defined context and range list to calculate the health index
    col_definition=pd.read_csv("DataPreProcessing/Column_Definition.csv")
    # calculate the total score
    total_score=10* col_definition.shape[0] 
    col_definition=col_definition.values
    score=0
    for col in col_definition:
        temp=0
        missing=missing_rate[col[0]]
        if(col[1]=="No" or col[1]=="Yes"):
            score+=10*(1-missing)
            if(str(col[2])!='nan' and str(col[3])!='nan'):
                temp=my_data[(my_data[col[0]]>=int(col[2])) & \
                                   (my_data[col[0]]<=int(col[3]))].shape[0]
                temp=int(my_data.shape[0]*(1-missing)-temp)*0.5
                print("There are "+str(temp*2)+" invalid values in "+col[0]+" column")
        if(temp>10):
            temp=10
        score-=temp
    print("\n\n============================================================\n\n")
    print("Perform some manual validity check based on the understanding of data: ")
    #Additional mannual check based on the understandng of dataset
    temp=my_data[my_data['price']>'$$$$'].shape[0]
    print("There are "+str(temp)+" invalid values column price")
    if(temp>10):
        temp=10
    score-=temp*0.5    
    temp=my_data[my_data['zipcode']<10000].shape[0]
    print("There are "+str(temp)+" invalid values column zipcode")
    if(temp>10):
        temp=10
    score-=temp*0.5
    
    cleaness_index=score/total_score
    return cleaness_index

my_data=pd.read_csv("RawData/Resturant_Full_Infomation.csv", sep=',')
print("The overall health score of the dataset is "+str(DataCleanessCheck(my_data)))
 
    
