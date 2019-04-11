#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:02:47 2018

@author: Shaoyu Feng
"""

#%%
import pandas as pd 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
from scipy.stats import mode
import seaborn as sns
import numpy as np
from sklearn import decomposition
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest
from scipy import stats



# =============================================================================
# Data Clearning (To Deal with Missing Data)
# =============================================================================

def drop_context(data):
    """
    This is the part to remove any duplicated context columns 
    Remove Meaningless/Duplicated features based on Heusistics
    Remove context data not in use in later analytics 
    """
    
    data=data.drop(['url','id','Name','category_y','lowprice',
                    'Zipcode','is_closed','Mean travel time to work (commute)', 
                    'Never married','Now married', 'Separated',
                    'Widowed', 'Divorced','Two or more races population',
                    'Caters','transactions','star1','star2','star3',
                    "Native Hawaiian and Other Pacific Islander population",
                    "Some other race population",
                    'star4','star5','highprice'],axis=1)
    return data

def basic_stats_generation(data):
    """
    To Generate Basic Statistics of Input Data
    """
    
    # To Find basic statistics includes max, min, median, standard deviation 
    # For all numerical data 
    basic_statistics=data.describe()
    basic_statistics.to_csv('DataCleaning/basic_statistics.csv')
    # Find mode for categorical Data
    mode_data=data.mode()
    mode_data.to_csv('DataCleaning/mode.csv')
    

def drop_high_missing_columns(data, thre):
    """
    To Drop Features with large portion of missing values
    """
    
    # Find out missing rate for 
    missing_rate=data.isnull().sum()/data.shape[0]
    print(missing_rate)
    # Will Drop All Features if missing rate is >0.5
    # Since Any imputation will cause bias in the data 
    drop_columns=list(missing_rate[missing_rate>=thre].keys())
    data=data.drop(drop_columns,axis=1)
    return data


def missing_data_handling (data):
    """
    To massively deal with missing data in the columns 
    
    """
    
    # Dealing with missing Data
    missing_rate=data.isnull().sum()/data.shape[0]
    columns=missing_rate[missing_rate>0]
    print(columns)
    
    # For Categorical Columns, we will imputate the missing values by mode 
    categorical_cols=['Accept_Credit_Card', 'Alcohol',
           'Outdoor_Seating', 'Parking', 'Take_out', 'Takes_Reservations', 'WIFI',
           'Ambience', 'Attire', 'Noise_Level']
    mode_data=data.mode()
    for name in categorical_cols:
        data[name].fillna(mode_data[name][0],inplace=True)
    
    # For Numerical Colymns, will imputate the missing value by median 
    # Due to large variance in the data 
    numerical_cols=['atm', 'bank', 'bar', 'beauty_salon',
           'bus_station', 'cafe', 'gym', 'school', 'Zip code population in 2016',
           'White population', 'Black population', 'American Indian population',
           'Asian population',
           'Hispanic or Latino population','High school or higher',
       'Graduate or professional degree', 'Unemployed']
    meadian_statistics=data.median()
    
    for name in numerical_cols:
        data[name].fillna(meadian_statistics[name],inplace=True)
    return data

# =============================================================================
# Feature Transformation and Engineering 
# =============================================================================
# Demographical informatio 

def demography_cal(row):
    """
    To Calculate the percentage of each demography 
    """
    total=row['Zip code population in 2016']       
    white=row['White population']                   
    black=row['Black population']                   
    indian=row['American Indian population']         
    asian=row['Asian population']                   
    his_lat=row['Hispanic or Latino population']
    return pd.Series([white/total,black/total,
                      indian/total,asian/total,his_lat/total])

def average_price(row):
    """
    To translate the categorical price featue in to number 
    Based on the infomation provided by 
    """
    price=row['price']
    if (len(price)==1):
        return 10
    elif (len(price)==2):
        return 20
    elif(len(price)==3):
        return 45
    elif(len(price)==4):
        return 60
    
def binary_transformation(row):
    """
    To transform some of the binary feature in to 0 and 1 
    For the ease of future analysis 
    """
    Accept_Credit_Card=1 if (row['Accept_Credit_Card']=='Yes') else 0
    Outdoor_Seating= 1 if (row['Outdoor_Seating']=='Yes') else 0
    Take_out=1 if (row['Take_out']=='Yes') else 0
    Takes_Reservations=1 if (row['Takes_Reservations']=='Yes') else 0
    wifi=1 if (row['WIFI']=='FREE') else 0 
    Noise_Level=row['Noise_Level']
    if(Noise_Level=='Quite'):
        Noise_Level=1
    elif (Noise_Level=='Average'):
        Noise_Level=2
    elif (Noise_Level=='Loud'): 
        Noise_Level=3
    else:
        Noise_Level=4
    return pd.Series([Accept_Credit_Card,Outdoor_Seating,Take_out,
                      Takes_Reservations,wifi,Noise_Level])

def zip_code_handling(row):
    zipcode=row['zipcode']
    if np.isnan(zipcode):
        zipcode = np.nan_to_num(zipcode)
    zipcode=str(int(zipcode))
    if(len(zipcode)<5):
        zipcode='0'+zipcode
    return str(zipcode)
    

def Additional_Feature_Engineering(data):
    """
    This is a wrapper function to perform additional feature Engineering 
    """
    # Additional Feature Engineering 
    # Calculat ratios in demography data
    data[['White population','Black population','American Indian population',
         'Asian population','Hispanic or Latino population']] \
          = data.apply(lambda row: demography_cal(row), axis=1)
    data=data.drop(['Zip code population in 2016'],axis=1)
    # Add average price based on the infomrtion provided by YELP 
    data['average_price']=data.apply(lambda row: average_price(row), axis=1)
    # Dealing with zip code leading zeros issues
    data['zipcode']=data.apply(lambda row: zip_code_handling(row),axis=1)
    # To transform some binarly feature into 0 and 1 for the ease of analutics 
    data[['Accept_Credit_Card','Outdoor_Seating','Take_out',
                          'Takes_Reservations','WIFI','Noise_Level']] \
          = data.apply(lambda row: binary_transformation(row), axis=1)
    
    return data

# =============================================================================
# Data Binning
# =============================================================================

def review_count_bining(data):
    names2 = ['Small No Reviews','Medium No Reviews',
              'Meadium to Large Reviews', 'large Number of Reviews'
              ,'Very Popular Resturant']
    bins = [-1, 150, 300, 600,1000,9999999]
    data['review_count_binned'] = pd.cut(data['review_count'], 
        bins,labels=names2)
    
    fig=plt.figure(1,figsize=(13, 10))
    plt.subplot(2,2,1)
    data['review_count'].plot(kind='hist',
        title='Review Counts Before Bining',bins=50)               
    plt.subplot(2,2,2)
    data['review_count_binned'].value_counts().plot(kind='bar',
        title='Review Counts ater Bining')
    fig.suptitle('Comparison before and after bining for review count ', 
                 fontsize=20)
    plt.show()
    fig.savefig('DataCleaning/Bining_Comparasion.jpg')
    
    fig.clear()
    plt.close()
    return data

# =============================================================================
# Outlier Detection
# =============================================================================


def LOF_outlier_detection (data):
    """
    Use LOF methods to detect outlier in multidimensional space 
    Use PCA to visualize results 
    """
    X=data[['atm', 'bank', 'bar', 'bus_station', 'cafe' ]].values   
    clf = LocalOutlierFactor(n_neighbors=40)
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    y_pred = clf.fit_predict(X)
    X_scores = clf.negative_outlier_factor_      
    # Use PCA for better visualization.
    pca2D = decomposition.PCA(2)
    #Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(X)
    
    plt.title("Local Outlier Factor (LOF)")
    plt.scatter(plot_columns[:, 0],plot_columns[:, 1], color='blue', s=3., label='Data points')
    # Plot using a scatter plot and shade by cluster label
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    
    plt.scatter(plot_columns[:, 0],plot_columns[:, 1], s=500 * radius, edgecolors='r',
                facecolors='none', label='Outlier scores')
    plt.axis('tight')
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.savefig('DataCleaning/LOF_outlier_Detection.jpg')
    max_val = radius.max()
    index_max = list(radius).index(max_val)
    return index_max


def isolation_forest(data):
    """
    Isolation Forest for outlier detection
    """
    
    fig,ax= plt.subplots(figsize=(6,4))
    X=data[['atm', 'bank', 'bar', 'bus_station', 'cafe' ]]
    pca2D = decomposition.PCA(2)
    #Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(X)
    def plot_model(lables, alg_name):
        # plt.figure(plot_index)
        color_code = {'anomaly':'red', 'normal':'green'}
        colors = [color_code[x] for x in labels]
        ax.scatter(plot_columns[:,0], plot_columns[:,1], color=colors, marker='.', label='red = anomaly')
        ax.legend(loc="lower right")
    
        leg = plt.gca().get_legend()
        leg.legendHandles[0].set_color('red')
        ax.set_title(alg_name)
        plt.show()
    
    outliers_fraction = 0.001
    model = IsolationForest().fit(X)
    scores_pred = model.decision_function(X)
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
    labels = [('anomaly' if x<threshold else 'normal') for x in scores_pred]
    plot_model(labels, 'Isolation Forest')
    fig.savefig('DataCleaning/Isolation_Forest.jpg')
    fig.clear()
    plt.close()
    
def basic_outlier_detection(data):
    """
    To use Boxplot or Z-Score for outlier detection 
    """
    
    # Use Box Plot for Outlier Detection 
    sns.boxplot(data=data[['atm', 'bank', 'bar', 'bus_station', 'cafe' ]],
                palette="Set1")
    plt.title('BoxPlot for Outlier Detection')
    plt.savefig('DataCleaning/Boxplot.jpg')
    plt.show()
    plt.close()
    # Use Z Score to identify outlier 
    
    z = np.abs(stats.zscore(data[['atm', 'bank', 'bar', 'bus_station', 'cafe' ]]))
    threshold = 4
    print(np.where(z > threshold))

#%%
    

def main():
    
    data=pd.read_csv('RawData/Full_Information.csv',index_col=0)
    # Context Data Drop 
    data=drop_context(data)
    basic_stats_generation(data)
    # Deal with Missing Data
    data=drop_high_missing_columns(data, 0.3)
    print(data.columns)
    try:
        data=missing_data_handling (data)
    except:
        pass
    # Additional Feature Engineering 
    data=Additional_Feature_Engineering(data)
    # Data Binning
    data=review_count_bining(data)
    # Outlier Detection and handling 
    outlier_index=LOF_outlier_detection(data)
    isolation_forest(data)
    data=data.drop(data.index[[outlier_index]])
    
    # Output Cleaned DataSet.
    #data.to_csv('RawData/Full_Information_Cleaned.csv',index=False)

if __name__ == "__main__":
	 main()

















