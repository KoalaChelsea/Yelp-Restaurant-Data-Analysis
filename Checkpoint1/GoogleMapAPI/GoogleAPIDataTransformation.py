#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:32:04 2018
This is the script to pivot the google map API data
@author: Shaoyu
"""


#%%

import glob 
import pandas as pd 
#%%
filelist=glob.glob("GoogleMapAPI/*.csv")
for data_file in filelist: 
    filename=data_file.split("/")[1]
    print(filename)
    map_data=pd.read_csv(data_file,sep=',')
    my_df=map_data[['yelp_id','place_id','types']]
    my_df=my_df.pivot_table(values='place_id', index='yelp_id', columns='types', aggfunc=pd.Series.nunique)
    my_df.to_csv("RawData/GoogleMapAPI/"+filename)
