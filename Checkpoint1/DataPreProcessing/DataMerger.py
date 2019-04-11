#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:29:32 2018
q
@author: Shaoyu
"""
#%%
import pandas as pd
import glob

filelist=glob.glob("RawData/DataPulling/*.csv")
combined_csv1 = pd.concat([pd.read_csv(f) for f in filelist ])
combined_csv1.to_csv("RawData/Resturant_API_Combined.csv",index=None)

filelist=glob.glob("RawData/DataScrape/*.csv")
combined_csv2 = pd.concat([pd.read_csv(f) for f in filelist ])
combined_csv2.to_csv("RawData/Resturant_Scrape_Combined.csv",index=None)
filelist=glob.glob("RawData/GoogleMapAPI/*.csv")
combined_csv3 = pd.concat([pd.read_csv(f) for f in filelist ])
combined_csv3.to_csv("RawData/Resturant_Map_API_Combined.csv",index=None)
final_df=pd.merge(combined_csv1,combined_csv2, how='left', left_on='id', right_on='Id')
final_df=pd.merge(final_df,combined_csv3, how='left', left_on='id', right_on='yelp_id')
final_df=final_df.drop_duplicates(['id'])
final_df=final_df.drop(['Id','yelp_id'], axis=1)
final_df.to_csv("RawData/Resturant_Full_Infomation.csv",index=None)
