#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 09:23:23 2018

@author: Shaoyu
"""

from yelpapi import YelpAPI
import pandas as pd
import numpy as np 
from lxml import html  
import requests
import time 
import json
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import os
import math

#%%

# =============================================================================
# Function to make use of Yelp API to get resturant infomation
# =============================================================================
def yelpAPITesting():
    api_key="1kB3IeEvGUKEtQqyFg9cuVGztBPX-kJAHnSFNRSEKSSjvetFfxvp8GtmSpdwcn0dF7soebTxhiG-wclEIqRy7D3cjKWlrABZ10XnICV17TuqYEVq4lpVMyFDbiqxW3Yx"
    yelp_api = YelpAPI(api_key)
    neighbor_list=['georgetown']
    resultslist=[]
    
    for neighbor in neighbor_list:
        loc=(neighbor.replace("\n","").lower())+", washington, dc"
        for count in range (5):
            try:
                response = yelp_api.search_query(term='resturant', \
                                         location=loc,\
                                         sort_by='distance',\
                                         radius=20000,\
                                         offset=count*30,\
                                         limit=50)
                resultslist.append(response)
            except:
                continue
    col_names=['name','latitude','longitude','is_closed','zipcode',\
           'city','state','price','rating','url','review_count', \
           'transactions','category','id']
    my_df  = pd.DataFrame(columns = col_names)
    for queryresults in resultslist:
        for temp in queryresults['businesses']:
            try:
                res_id=temp['id']
                categories=temp['categories']
                category=""
                for cat in categories:
                    category+= cat['alias']+","
                category=category[0:-1]
                name=temp['name']
                lat=temp['coordinates']['latitude']
                long=temp['coordinates']['longitude']
                is_closed=temp['is_closed']
                zipcode=temp['location']['zip_code']
                city=temp['location']['city']
                state=temp['location']['state']
                price=temp['price']
                rating=temp['rating']
                url=temp['url']
                review_count= temp['review_count']
                transactions=temp['transactions']
                my_df=my_df.append({col_names[0]:name,\
                                      col_names[1]:lat,\
                                      col_names[2]:long ,\
                                      col_names[3]:is_closed, \
                                      col_names[4]:zipcode ,\
                                      col_names[5]:city, \
                                      col_names[6]:state, \
                                      col_names[7]:price,\
                                      col_names[8]:rating,\
                                      col_names[9]:url, \
                                      col_names[10]:review_count ,\
                                      col_names[11]:transactions, \
                                      col_names[12]:category, \
                                      col_names[13]:res_id
                                      }, ignore_index=True)
            except:
                continue
    
        my_df=my_df.drop_duplicates(['id'])
        print("SampleData from Yelp API: ")
        print("=========================================================")
        print(my_df.head(10))
        print("\n\n\n")
        return my_df
# =============================================================================
# Function to make use of Web Scrape to get additional Resturant infomation
# ============================================================================= 
def queryAdditionalInfo(res_id, res_url):
    col_names=['Id','Name','category','lowprice','highprice','health_index',\
           'star1','star2','star3','star4','star5','Accept_Credit_Card', \
           'Alcohol','Appointment_Only','Caters','Delivers','Dogs_Allowed','Outdoor_Seating', \
           'Parking','Smoking_allowed','Take_out','Takes_Reservations','Wheelchair_Accessible','WIFI', \
           'Opened_24hrs','Ambience','Attire','Noise_Level','Music']
    my_df  = pd.DataFrame(columns = col_names)
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
    print("SampleData from Yelp WebScrape: ")
    print("=========================================================")
    print(my_df)
    print("\n\n\n")
    
# =============================================================================
# Function to Make use of GoogleMapAPI to Generate Resturant Nearby infomrtion 
# =============================================================================    
    
def GoogleMapAPI(restaurant_file):
    location_data = restaurant_file[['latitude','longitude']]
    yelp_id = restaurant_file['id']
    lat_lng_list = list()
    for i in np.arange(len(location_data)):
        lat_lng_list = lat_lng_list + [(str(location_data['latitude'][i])+','+str(location_data['longitude'][i]))]
    
    BaseURL ='https://maps.googleapis.com/maps/api/place/nearbysearch/json?'
    type_list = ['bus_station','subway_station','taxi_stand','supermarket','shopping_mall','school',
                 'book_store','museum','atm','bank','train_station','gym','gas_station','cafe','bar',
                 'beauty_salon','movie_theater']
    columns_name = ['yelp_id','restaurant_location','id','place_id','types','name','rating','geometry','plus_code','vicinity']
    test=False
    output_all = pd.DataFrame(columns = columns_name)
    time.sleep(3)
    count = 0
    output_all=pd.read_csv("GoogleMapAPI/washington_restaurant_external.csv")
    if test:
        for lat_lng in lat_lng_list:
            count = count+1
            for types in type_list:
                #bus_station
                URLPost = {'location':lat_lng,
                           'radius':'500',
                           'type':types,
                           'key':'AIzaSyBr1aciMB7n6Exh-3gDwCudczKAAvbtvT8'
                           }
        
                response=requests.get(BaseURL, URLPost) #get data from the API
                jsontxt = response.json() #save the data as json format
                combine = jsontxt['results']
                for i in np.arange(len(combine)):
                    combine[i]['geometry'] = str(list(combine[i]['geometry']['location'].values())[0])+','+ str(list(combine[i]['geometry']['location'].values())[1])
                    try:
                        combine[i]['plus_code'] = combine[i]['plus_code']['compound_code']
                    except:
                        combine[i]['plus_code'] = 'No information'
                    try:
                        combine[i]['rating'] = combine[i]['rating']
                    except:
                        combine[i]['rating'] = 'No information'
                    
                    #combine[i]['types'] = combine[i]['types'][0]
                    combine[i]['types'] = types
                
                output = pd.DataFrame(columns = columns_name)
            
                for j in np.arange(len(columns_name)-2):
                    j = j+2
                    col = columns_name[j]
                    hold = list()
                    for i in np.arange(len(combine)):
                        hold = hold + [combine[i][col]]
                
                    output[col] = hold
                
                output['restaurant_location'] = np.repeat(lat_lng,len(output))
                output['yelp_id'] = np.repeat(yelp_id[count-1],len(output))
                output_all = output_all.append(output)
    print("SampleData from Google Map API: ")
    print("=========================================================")
    print(output_all.head(100))    
    print("\n\n\n")
    return output_all.head(100)
# =============================================================================
# Function to perform a cleaness Check 
# =============================================================================
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
    total_score=10* col_definition[col_definition['Context']=="No"].shape[0] 
    col_definition=col_definition.values
    score=0
    for col in col_definition:
        temp=0
        missing=missing_rate[col[0]]
        if(col[1]=="No"):
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

# =============================================================================
# Feature Generation
# =============================================================================

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
my_df=yelpAPITesting()
input_data=my_df.head(1)[['url','id']].values.tolist()
for input in input_data:
    queryAdditionalInfo(input[1], input[0])
    time.sleep(2)
#%%    
my_df=GoogleMapAPI(my_df.head(10))
my_df=my_df[['yelp_id','place_id','types']]
my_df=my_df.pivot_table(values='place_id', index='yelp_id', columns='types', aggfunc=pd.Series.nunique)
print("SampleData from Google Map API after transformation: ")
print("=========================================================")
print(my_df.head(10))    
print("\n\n\n")

print("=========================================================")
print("To Perform out data cleaness check and feature generation")
my_data=pd.read_csv("RawData/Resturant_Full_Infomation.csv", sep=',')
print("The overall health score of the dataset is "+str(DataCleanessCheck(my_data)))

print("=========================================================")
print("To Perform data feature generation")
first_feature(my_data)
print("New Feature 1 and counts:")
print(my_data['popularity'].value_counts(dropna = False))
my_data['price_category']=my_data.apply (lambda row: second_feature(row),axis=1)
print("New Feature 2 and counts:")
print(my_data['price_category'].value_counts(dropna = False))
print("New Feature 3 and counts:")
my_data['transportation']=my_data.apply (lambda row: third_feature(row),axis=1)
print(my_data['transportation'].value_counts(dropna = False))






