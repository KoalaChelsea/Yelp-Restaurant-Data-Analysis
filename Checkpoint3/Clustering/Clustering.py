#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:41:37 2018

@author: Jiaxuan Sun
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pylab as pl
from pprint import pprint
from sklearn.metrics import silhouette_score,calinski_harabaz_score
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
#%%

def data_processing():
    # Read data frame
    myData = pd.read_csv('RawData/Full_Information_Cleaned.csv', sep = ',',encoding = 'utf-8-sig')
    
    city_list = myData['city'].unique()
    i = 0
    myData['city_numeric'] = np.repeat(0,len(myData['city']))
    for city in city_list:
        i = i+1
        myData['city_numeric'][myData['city'] == city] = i
    
    state_list = myData['state'].unique()
    i = 0
    myData['state_numeric'] = np.repeat(0,len(myData['state']))
    for state in state_list:
        i = i+1
        myData['state_numeric'][myData['state'] == state] = i
    
    price_list = myData['price'].unique()
    myData['price_numeric'] = np.repeat(0,len(myData['price']))
    for price in price_list:
        myData['price_numeric'][myData['price'] == price] = len(price)
    
    #richness
    category_x_list = myData['category_x'].unique()
    myData['category_x_numeric'] = np.repeat(0,len(myData['category_x']))
    for category_x in category_x_list:
        myData['category_x_numeric'][myData['category_x'] == category_x] = len(category_x.split(','))
    
    
    
    #richness
    myData['Alcohol_numeric'] = np.repeat(0,len(myData['Alcohol']))
    myData['Alcohol_numeric'][myData['Alcohol'] == 'No'] = 1
    myData['Alcohol_numeric'][myData['Alcohol'] == 'Beer & Wine Only'] = 2
    myData['Alcohol_numeric'][myData['Alcohol'] == 'Full Bar'] = 3
    
    
    
    #richness
    Parking_list = myData['Parking'].unique()
    myData['Parking_numeric'] = np.repeat(0,len(myData['Parking']))
    for Parking in Parking_list:
        myData['Parking_numeric'][myData['Parking'] == Parking] = len(Parking.split(','))
        
    '''
    #richness
    Ambience_list = myData['Ambience'].unique()
    myData['Ambience_numeric'] = np.repeat(0,len(myData['Ambience']))
    for Ambience in Ambience_list:
        myData['Ambience_numeric'][myData['Ambience'] == Ambience] = len(Ambience.split(','))
    '''
    
    myData['Unemployed'] = myData['Unemployed'].map(lambda x: x * 10 ) 
    
    numericDF = myData.select_dtypes(include = np.number) 
    myData = pd.DataFrame(numericDF, columns = numericDF.columns)
    
    
    return (myData)


def normalize_data(myData):
    # Select only numeric columns
    mydata = preprocessing.normalize(myData,axis = 1)

    return(mydata)



def clustering(myData,method,name,cat):	
    #record_performance = pd.DataFrame(columns = ['method','category','n','score'])
    
    cluster_labels = method.fit_predict(myData)
    # Determine if the clustering is good
    # silhouette_avg = silhouette_score(myData, cluster_labels)
    calinski_avg = calinski_harabaz_score(myData, cluster_labels)
    print("The average calinski_harabaz_score is :", calinski_avg)  
    
    silhouette_avg = silhouette_score(myData, cluster_labels)
    print("The average silhouette_score is :", silhouette_avg)
    
    # Use PCA
    # Turn the data into two columns with PCA
    plot_columns = PCA(n_components=2).fit_transform(myData)
    # Plot using a scatter plot and shade by cluster label
    plt.figure()
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    titlename = "Clustering/Cluster Plot with " + name + " method for "+ cat
    plt.title(titlename)
    plt.savefig(titlename)
    plt.show()
    
    
        
    # Plot a 3D graph for better visualization
    # The code is adapted from 
    # http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
    # #sphx-glr-auto-examples-datasets-plot-iris-dataset-py
    fig = plt.figure(figsize=(10, 7.5))
    ax = Axes3D(fig, elev=-165, azim=115)
    X_reduced = PCA(n_components=3).fit_transform(myData)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=cluster_labels,
               cmap=plt.cm.Set1, edgecolor='k', s=40)    
    titlename2 = "Clustering/First three PCA directions - "+ name + ' for '+cat
    ax.set_title(titlename2)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.savefig(titlename2)
    plt.show()
    
def control(cat,myData_sub):
    if cat =='Internal Factors':
        n = 2
    elif cat =='Internal Richness':
        n = 3
    elif (cat =='Population Composition' or cat =='Education Level'):
        n = 4
    else:
        n = len(np.unique(myData['rating']))
        
    
    # Models
    Hierarchical = AgglomerativeClustering(affinity = 'euclidean', compute_full_tree = 'auto',
                                           connectivity = None, linkage = 'ward', memory = None, n_clusters=n,
                                           pooling_func='deprecated')
    kmeans = KMeans(n_clusters = n)
    DbScan = DBSCAN(eps=0.1, min_samples=130)
        
    # Run models
        
    clustering(myData_sub,kmeans,"K-means",cat)
    clustering(myData_sub,Hierarchical,"Hierarchical",cat)
        
    choose_n = 'y'
    while choose_n == 'y':
        choose_n = input('Do you want to change the value of n(y/n): ')
        if choose_n == 'y':
            n = int(input('Please input the new value for n: '))
            Hierarchical = AgglomerativeClustering(affinity = 'euclidean', compute_full_tree = 'auto',
                                                   connectivity = None, linkage = 'ward', memory = None, n_clusters=n,
                                                   pooling_func='deprecated')
            kmeans = KMeans(n_clusters = n)
            clustering(myData_sub,kmeans,"K-means",cat)
            clustering(myData_sub,Hierarchical,"Hierarchical",cat)
            
        
        
    clustering(myData_sub,DbScan,"DBScan",cat)
    choose_eps = 'y'
    while choose_eps == 'y':
        choose_eps = input('Do you want to change the value of eps/min_sample: ')
        if choose_eps == 'y':    
            input_eps = float(input('Please input the new value for eps: '))
            input_min_sample = int(input('Please input the new value for min_sample: '))
            DbScan = DBSCAN(eps=input_eps, min_samples=input_min_sample) 
            clustering(myData_sub,DbScan,"DBScan",cat)
            

#%%


myData = data_processing() #normalize and only keep numeric columns
# Initial the number of clusters as 6

columns = myData.columns

#for Population composition(4)
myData_sub = myData[columns[19:24]]
myData_sub = normalize_data(myData_sub)
cat = 'Population Composition'
control(cat,myData_sub)


# Comment for the ease of testing

# =============================================================================
# #for neighborhood(8)
# myData_sub = myData[columns[11:19]]
# myData_sub = normalize_data(myData_sub)
# cat = 'Neighborhood'
# control(cat,myData_sub)
# 
# 
# #for Internal Factors (2)
# myData_sub = myData[columns[5:10]]
# myData_sub = normalize_data(myData_sub)
# cat = 'Internal Factors'
# control(cat,myData_sub)
# 
# 
# #for Internal Richness(3)
# myData_sub = myData[columns[31:34]]
# myData_sub = normalize_data(myData_sub)
# cat = 'Internal Richness'
# control(cat,myData_sub)
#         
#    
# #for Education Level(4)
# myData_sub = myData[columns[24:27]]
# myData_sub = normalize_data(myData_sub)
# cat = 'Education Level'
# control(cat,myData_sub)
# =============================================================================

#if __name__ == "__main__":
 #   main() 
    
    
    
   
    