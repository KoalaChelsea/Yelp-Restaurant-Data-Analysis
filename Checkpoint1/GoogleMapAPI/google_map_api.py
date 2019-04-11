#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 23:33:59 2018

@author: stephensun
"""

import requests
import pandas as pd
import numpy as np
import os
import time

def restaurant_location(file):
    location_data = file[['latitude','longitude']]
    lat_lng_list = list()
    for i in np.arange(len(location_data)):
        lat_lng_list = lat_lng_list + [(str(location_data['latitude'][i])+','+str(location_data['longitude'][i]))]

    return(lat_lng_list)
    
    

def restaurant_id(file):
    yelp_id = file['id']
    return(yelp_id)


def get_data(lat_lng_list,type_list,columns_name):
    output_all = pd.DataFrame(columns = columns_name )
    BaseURL ='https://maps.googleapis.com/maps/api/place/nearbysearch/json?'
    count = 0
    
    for lat_lng in lat_lng_list:
        count = count+1
        for types in type_list:
            #bus_station
            URLPost = {'location':lat_lng,
                       'radius':'500',
                       'type':types,
                       'key':'AIzaSyCcIKAbu-8KsoTbgqZagVdgseHR-loeJGQ'
                       }

            response=requests.get(BaseURL, URLPost) #get data from the API
            jsontxt = response.json() #save the data as json format
    
            combine = jsontxt['results']
    
            #second_time query
            if len(jsontxt) == 4:
                time.sleep(1.5) #set a time delay
    
                while(True):
                    #get addition results
                    additional_result = jsontxt['next_page_token']
                    URLPost_additional={'pagetoken':additional_result,
                                        'key':'AIzaSyCcIKAbu-8KsoTbgqZagVdgseHR-loeJGQ'}

                    response_additional= requests.get(BaseURL, URLPost_additional) #get data from the API
                    jsontxt_additional = response_additional.json() #save the data as json format

                    status = jsontxt_additional['status']
        
                    if status == 'INVALID_REQUEST':
                        print ('Need a time delay, try again')
                    else:
                        print('The second time query succeed!')
                        combine = combine + jsontxt_additional['results']
                        break
     
                #third_time query
                if len(jsontxt_additional) == 4:
                    print('Need the third time query!')
                    time.sleep(1.5) #set a time delay
    
                    while(True):
                        additional_result_third = jsontxt_additional['next_page_token']
                        URLPost_additional_third={'pagetoken':additional_result_third,
                                                  'key':'AIzaSyCcIKAbu-8KsoTbgqZagVdgseHR-loeJGQ'}

                        response_additional_third= requests.get(BaseURL, URLPost_additional_third) #get data from the API
                        jsontxt_additional_third = response_additional_third.json() #save the data as json format
                
                        status = jsontxt_additional_third['status']
                
                        if status == 'INVALID_REQUEST':
                            print ('Need a time delay, try again') 

                        else:
                            print('The third time query succeed!')
                            combine = combine + jsontxt_additional_third['results']
            
                            if len(jsontxt_additional_third) == 4:
                                print('The data volume is out of 60.')
                                break
                            else:
                                break

            for i in np.arange(len(combine)):
                combine[i]['geometry'] = str(list(combine[i]['geometry']['location'].values())[0])+','+ str(list(combine[i]['geometry']['location'].values())[1])
                try:
                    combine[i]['plus_code'] = combine[i]['plus_code']['compound_code']
                except:
                    combine[i]['plus_code'] = 'No information'
                try:Phase1Testing
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
        
        
        print(lat_lng+ ' was finished, remain '+str(len(lat_lng_list)-count)+' need to be done.')    
    
    return(output_all)

if __name__ == '__main__': 
    path = os.getcwd()

    #########################change this###########################
    restaurant_file = pd.read_csv(path+'/Boston_Resturant.csv',nrows=1) 
    #########################change this###########################
    
    lat_lng_list = restaurant_location(restaurant_file)    
    yelp_id = restaurant_id(restaurant_file)
    type_list = ['bus_station','subway_station','taxi_stand','supermarket','shopping_mall','school',
                 'book_store','museum','atm','bank','train_station','gym','gas_station','cafe','bar',
                 'beauty_salon','movie_theater']
    columns_name = ['yelp_id','restaurant_location','id','place_id','types','name','rating','geometry','plus_code','vicinity']

    

    output_all = get_data(lat_lng_list,type_list,columns_name)
    
    #########################change this###########################
    output_all.to_csv(path+'/Boston_Resturant_external.csv',sep = ',',index = False)
    #########################change this###########################
    
       


