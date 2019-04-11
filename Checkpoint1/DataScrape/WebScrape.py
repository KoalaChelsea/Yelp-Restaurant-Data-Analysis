#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:59:06 2018

@author: Shaoyu
"""
#%%
from lxml import html  
import requests
import time 
import json
import pandas as pd
#%%
def queryAdditionalInfo(res_id, res_url,my_df):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
    response = requests.get(res_url, headers=headers, verify=False).text
    parser = html.fromstring(response)
    raw_name = parser.xpath("//h1[contains(@class,'page-title')]//text()")
    raw_category  = parser.xpath('//div[contains(@class,"biz-page-header")]//span[@class="category-str-list"]//a/text()')
    details_table = parser.xpath("//div[@class='short-def-list']//dl")
    raw_price_range = parser.xpath("//dd[contains(@class,'price-description')]//text()")
    raw_health_rating = parser.xpath("//dd[contains(@class,'health-score-description')]//text()")
    rating_histogram = parser.xpath("//table[contains(@class,'histogram')]//tr[contains(@class,'histogram_row')]")
    # to reorder the necessary infomation 
    info = {}
    for details in details_table:
        raw_description_key = details.xpath('.//dt//text()')
        raw_description_value = details.xpath('.//dd//text()')
        description_key = ''.join(raw_description_key).strip()
        description_value = ''.join(raw_description_value).strip()
        info[description_key]=description_value
        json_data = json.dumps(info)
                
    ratings_histogram ={} 
    for ratings in rating_histogram:
        raw_rating_key = ratings.xpath(".//th//text()")
        raw_rating_value = ratings.xpath(".//td[@class='histogram_count']//text()")
        rating_key = ''.join(raw_rating_key).strip()
        rating_value = ''.join(raw_rating_value).strip()
        ratings_histogram[rating_key]=rating_value
        json_data = json.dumps(ratings_histogram)
    	
    name = ''.join(raw_name).strip()
    health_rating = ''.join(raw_health_rating)
    price_range = ''.join(raw_price_range).strip()
    category = ','.join(raw_category)
    try:
        health_index=health_rating
    except:
        health_index=""
    low_price=price_range
    high_price=price_range
    try:
        star1=int(ratings_histogram['1 star'])
    except:
        star1=0
    try:
        star2=int(ratings_histogram['2 stars'])
    except:
        star2=0
    try:
        star3=int(ratings_histogram['3 stars'])
    except:
        star3=0
    try:
        star4=int(ratings_histogram['4 stars'])
    except:
        star4=0
    try:
        star5=int(ratings_histogram['5 stars'])
    except:
        star5=0
    # =============================================================================
    # Accepts Credit Cards
    #  Alcohol
    #  Appointment Only
    #  Caters
    #  Delivers
    #  Dogs Allowed
    #  Outdoor Seating
    #  Parking
    #  Smoking Allowed
    #  Take-out
    #  Takes Reservations
    #  Wheelchair Accessible
    #  Wi-Fi
    #  Opened 24hrs
    #  Gender Neutral Bathrooms
    #  Ambience
    #  Attire
    #  Music
    #  Noise Level
    # =============================================================================
    try:
        Accepts_Credit_Cards=info['Accepts Credit Cards']
    except:
        Accepts_Credit_Cards=""
    
    try:
        Alcohol=info['Alcohol']
    except:
        Alcohol=""
    
    try:
        Appointment_Only=info['Appointment Only']
    except:
        Appointment_Only=""
    try:
        Caters=info['Caters']
    except:
        Caters=""
    try:
        Delivers=info['Delivers']
    except:
        Delivers=""
    try:
        Dogs_Allowed=info['Dogs Allowed']
    except:
        Dogs_Allowed=""
    try:
        Outdoor_Seating=info['Outdoor Seating']
    except:
        Outdoor_Seating=""
    try:
        Parking=info['Parking']
    except:
        Parking=""
    try:
        Smoking_Allowed=info['Smoking Allowed']
    except:
        Smoking_Allowed=""
    try:
        Take_out=info['Take-out']
    except:
        Take_out=""
    try:
        Takes_Reservations=info['Takes Reservations']
    except:
        Takes_Reservations=""
    try:
        Wheelchair_Accessible=info['Wheelchair Accessible']
    except:
        Wheelchair_Accessible=""
    try:
        WiFi=info['Wi-Fi']
    except:
        WiFi=""
    try:
        Opened_24hrs=info['Opened 24hrs']
    except:
       Opened_24hrs=""
    try:
        Ambience=info['Ambience']
    except:
        Ambience=""
    try:
        Attire=info['Attire']
    except:
       Attire=""
    try:
        Noise_Level=info['Noise Level']
    except:
        Noise_Level=""
    try:
        Music=info['Music']
    except:
        Music=""
    col_names=['Id','Name','category','lowprice','highprice','health_index',\
           'star1','star2','star3','star4','star5','Accept_Credit_Card', \
           'Alcohol','Appointment_Only','Caters','Delivers','Dogs_Allowed','Outdoor_Seating', \
           'Parking','Smoking_allowed','Take_out','Takes_Reservations','Wheelchair_Accessible','WIFI', \
           'Opened_24hrs','Ambience','Attire','Noise_Level','Music']
    my_df=my_df.append({col_names[0]:res_id,\
                                  col_names[1]:name,\
                                  col_names[2]:category ,\
                                  col_names[3]:low_price, \
                                  col_names[4]:high_price ,\
                                  col_names[5]:health_index, \
                                  col_names[6]:star1, \
                                  col_names[7]:star2,\
                                  col_names[8]:star3,\
                                  col_names[9]:star4, \
                                  col_names[10]:star5 ,\
                                  col_names[11]:Accepts_Credit_Cards, \
                                  col_names[12]:Alcohol, \
                                  col_names[13]:Appointment_Only,\
                                  col_names[14]:Caters,\
                                  col_names[15]:Delivers ,\
                                  col_names[16]:Dogs_Allowed, \
                                  col_names[17]:Outdoor_Seating ,\
                                  col_names[18]:Parking, \
                                  col_names[19]:Smoking_Allowed, \
                                  col_names[20]:Take_out,\
                                  col_names[21]:Takes_Reservations,\
                                  col_names[22]:Wheelchair_Accessible, \
                                  col_names[23]:WiFi ,\
                                  col_names[24]:Opened_24hrs, \
                                  col_names[25]:Ambience, \
                                  col_names[26]:Attire,\
                                  col_names[27]:Noise_Level,\
                                  col_names[28]:Music
                                  }, ignore_index=True)
    return my_df
    
    
#%%
col_names=['Id','Name','category','lowprice','highprice','health_index',\
           'star1','star2','star3','star4','star5','Accept_Credit_Card', \
           'Alcohol','Appointment_Only','Caters','Delivers','Dogs_Allowed','Outdoor_Seating', \
           'Parking','Smoking_allowed','Take_out','Takes_Reservations','Wheelchair_Accessible','WIFI', \
           'Opened_24hrs','Ambience','Attire','Noise_Level','Music']
my_df  = pd.DataFrame(columns = col_names)
my_df.to_csv(r'RawData/DataScrape/Huston_Resturant_Additional_Info.csv', index=None,sep=',', mode='w')
input_data=pd.read_csv("RawData/DataPulling/Huston_Resturant.csv")[['url','id']].values.tolist()
for input in input_data:
    my_df=queryAdditionalInfo(input[1], input[0],my_df)
    time.sleep(2)
my_df.to_csv(r'RawData/DataScrape/Huston_Resturant_Additional_Info.csv', index=None,sep=',', mode='a',header=None)







