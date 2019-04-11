#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 00:19:25 2018

@author: Jiaxuan Sun
"""
#%%
import numpy as np
import pandas as pd
import os
from apyori import apriori
#from efficient_apriori import apriori
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
#%%
def bin_columns(myData,columns_list):
    for i in np.arange(len(columns_list)):
        
        col = columns_list[i]
        
        low = np.percentile(myData[col],25)
        median = np.percentile(myData[col],50)
        high = np.percentile(myData[col],75)
       

        myData[col+'_bin'] = np.repeat(0,len(myData[col]))
        myData[col+'_bin'][myData[col] <= low] = 'Few_'+col
        myData[col+'_bin'][(low < myData[col]) & (myData[col]<= median)] = 'Moderate_'+col
        myData[col+'_bin'][(median < myData[col]) & (myData[col]<= high)] = 'Many_'+col
        myData[col+'_bin'][high < myData[col]] = 'Super many_'+col
    
    
    #bin noise_level column
    myData['Noise_Level_bin'] = np.repeat(0,len(myData['Noise_Level']))
    myData['Noise_Level_bin'][myData['Noise_Level'] == 2] = 'Low_Noise'
    myData['Noise_Level_bin'][myData['Noise_Level'] == 3] = 'Median_Noise'
    myData['Noise_Level_bin'][myData['Noise_Level'] == 4] = 'High_Noise'
    
    return (myData)


def rules_mining(myData_sub_list,support,confidence):
    association_rules = apriori(myData_sub_list,min_support = support,min_confidence = confidence )
    association_rules = list(association_rules)
    rules_dataframe = pd.DataFrame(columns = ['Rule','Transaction','Antecedent','Consequent','Support','Confidence','Lift'])
    print('The number of generated rules is:'+ str(len(association_rules)))
    row = 0
    
    for item in association_rules:    
        if len(item[0]) != 1:
            #pair = item[0]
            #items = [x for x in pair]
            #items = ','.join(items)
            #print("transaction: {" +items+'}')
            #print("Support: " + str(item[1]))
        
            for i in np.arange(len(item[2])):
                
                antecedent = item[2][i][0]
                antecedent = [x for x in antecedent]
                antecedent = ','.join(antecedent)
                
                consequent = item[2][i][1]
                consequent = [x for x in consequent][0]
                
                transaction = [antecedent,consequent]
                rule = "{" + antecedent + " -> " + consequent+'}'
                support_true = item[1]
                confidence_true = item[2][i][2]
                lift = item[2][i][3]
                
        
                #print("Rule: {" + antecedent + " -> " + consequent+'}')
                #print("Support: " + str(support))
                #print("Confidence: " + str(confidence))
                #print("Lift: " + str(lift))
                #print("=====================================")
                
                rules_dataframe.loc[row] = [rule,transaction,antecedent,consequent,support_true,confidence_true,lift]
                row = row+1
                print(str(row) +' rulse have generated!')
                
    rules_dataframe.to_csv('AssocationRule/'+str(support)+'_rules_dataframe.csv')            
    return (rules_dataframe)
        
def sep_cat(myData_sub_list):
    for i in np.arange(len(myData_sub_list)):
       cat = myData['category_x'][i].split(',')
       for j in np.arange(len(cat)):
           hold = cat[j]
           
           if hold in ['acaibowls','afghani','arabian']:
               hold = 'afghani'
           elif hold in ['australian','austrian']:
               hold = 'australian'
           elif hold in ['pubs','barcrawl','bars','beer_and_wine','beerbar','beergardens','brasseries',
                         'breweries','brewpubs','champagne_bars','cigarbars','cocktailbars','gaybars',
                         'hookah_bars','tikibars','whiskeybars','wine_bars','wineries','sportsbars','cocktailbars']:
               hold = 'bars'
           elif hold in ['internetcafe','cafes','cafeteria','coffee']:
               hold = 'cafes'
           elif hold in ['cantonese','chinese','szechuan','shanghainese','taiwanese']:
               hold = 'chinese'
           elif hold in ['chicken_wings','chickenshop']:
               hold = 'chicken'
           elif hold in ['caribbean','casinos']:
               hold = 'casinos'  
           elif hold in ['comedyclubs','danceclubs']:
               hold = 'comedyclubs'  
           elif hold in ['tapas','tapasmallplates','desserts','dimsum','candy','gelato','icecream']:
               hold = 'desserts' 
           elif hold in ['diners','dinnertheater']:
               hold = 'diners'
           elif hold in ['hotdog','hotdogs']:
               hold = 'hotdog'
           elif hold in ['fondue','hotpot']:
               hold = 'hotpot'
           elif hold in ['japanese','izakaya','japacurry','sushi']:
               hold = 'japanese'      
           elif hold in ['irish','irish_pubs']:
               hold = 'irish'
           elif hold in ['newamerican','newmexican','mexican','tradamerican']:
               hold = 'newmexican'   
           elif hold in ['noodles','pastashops','ramen']:
               hold = 'pastashops'       
           elif hold in ['vegan','vegetarian']:
               hold = 'vegetarianish'        
           elif hold in ['salad','sandwiches']:
               hold = 'sandwiches'     
           
            
           myData_sub_list[i] = myData_sub_list[i] + [hold]
    return (myData_sub_list)


def control(myData_sub_list):
    choose = 'y'
    while choose == 'y':
        choose = input('Do you want to change the support value and confidence value?(y/n)')
        if choose == 'y':
            support = float(input('Please input the min support value:'))
            confidence = float(input('Please input the min confidence value:'))
            rules_dataframe  = rules_mining(myData_sub_list,support,confidence)
            
            find_price = list(np.unique(myData_sub['rating']))
            find_popular = list(np.unique(myData_sub['review_count_binned']))
            find_useful = find_price + find_popular
            
            useful_rules = pd.DataFrame(columns = ['Rule','Transaction','Antecedent','Consequent','Support','Confidence','Lift'])
            
            for i in np.arange(len(rules_dataframe)):
                rules_transaction = rules_dataframe['Transaction'][i]
                set_c = set(rules_transaction) & set(find_useful)
                list_c = list(set_c)
                
                if len(list_c) != 0:
                    useful_rules.loc[i] = rules_dataframe.loc[i]
            
            useful_rules.to_csv('AssocationRule/'+str(support)+'_useful_rules.csv')



if __name__ == "__main__":
    
    myData = pd.read_csv('RawData/Full_Information_Cleaned.csv', sep = ',',encoding = 'utf-8-sig')
    
    myData['rating'] = myData['rating'].map(lambda x: str(x))
    myData['average_price'] = myData['average_price'].map(lambda x: str(x)) 
    columns_list = ['review_count','atm', 'bank', 'bar','beauty_salon', 'bus_station', 'cafe', 'gym', 'school',
                    'White population','Black population', 'American Indian population','Asian population', 
                    'Hispanic or Latino population','High school or higher', 'Graduate or professional degree'
                    ,'Unemployed']
    myData = bin_columns(myData,columns_list)
    columns = myData.columns
    
    #myData_sub = myData[['state','rating','review_count_bin','average_price']]  
    
    
    myData_sub = myData[['state','rating','review_count_binned','Noise_Level_bin',
                         'atm_bin', 'bank_bin', 'bar_bin', 'beauty_salon_bin', 'bus_station_bin', 
                         'cafe_bin', 'gym_bin', 'school_bin','White population_bin',
                         'Black population_bin', 'American Indian population_bin','Asian population_bin', 
                         'Hispanic or Latino population_bin','High school or higher_bin', 
                         'Graduate or professional degree_bin','average_price','Unemployed_bin']] 
    
    '''
    myData_sub = myData[['state','rating','review_count_binned','Noise_Level_bin',
                         'atm_bin', 'bank_bin', 'bar_bin', 'beauty_salon_bin', 'bus_station_bin', 
                         'cafe_bin', 'gym_bin', 'school_bin', 'average_price']] 
    
    '''

    myData_sub_array = np.array(myData_sub)
    myData_sub_list = myData_sub_array.tolist()

    #seperate the category_x
    myData_sub_list = sep_cat(myData_sub_list)
    
    #input the support and confidence and run the rules minning
    control(myData_sub_list)
            
   