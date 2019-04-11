#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 10:18:37 2018

@author: Shaoyu
"""
#%%
#To define the yelpapi object for later data pulling
from yelpapi import YelpAPI
import pandas as pd
import numpy as np 
api_key="1kB3IeEvGUKEtQqyFg9cuVGztBPX-kJAHnSFNRSEKSSjvetFfxvp8GtmSpdwcn0dF7soebTxhiG-wclEIqRy7D3cjKWlrABZ10XnICV17TuqYEVq4lpVMyFDbiqxW3Yx"
yelp_api = YelpAPI(api_key)
#%%
neighborhood=open('DataPulling/Huston_neighborhoods.txt','r')
neighbor_list=neighborhood.readlines()
resultslist=[]

for neighbor in neighbor_list:
    loc=(neighbor.replace("\n","").lower())+", huston"
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
#%%
col_names=['name','latitude','longitude','is_closed','zipcode',\
           'city','state','price','rating','url','review_count', \
           'transactions','category','id']
my_df  = pd.DataFrame(columns = col_names)
my_df.to_csv(r'RawData/DataPulling/Huston_Resturant.csv', index=None,sep=',', mode='w')

#%%

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
#my_df=my_df.drop_duplicates(['id'])
my_df=my_df.drop_duplicates(['id'])
my_df.to_csv(r'RawData/DataPulling/Huston_Resturant.csv', index=None, sep=',', mode='a',header=None)














