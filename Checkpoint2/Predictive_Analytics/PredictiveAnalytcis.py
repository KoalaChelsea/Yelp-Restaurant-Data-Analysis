#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 23:56:35 2018

@author: Shaoyu
"""

#%%
import pandas as pd 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import decomposition
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.linear_model import LogisticRegression
#%% 
# To Deal with the ratings 

def fusion_matrix(row,group_count):
    """
    Regenerate Label
    """
    
    rating=row['rating']
    base=rating-0.5
    review_count=row['review_count']
    basic_stats=group_count[rating]
    z_score=np.abs(review_count-basic_stats[0])/basic_stats[1]
    score=0
    if(z_score>1.5 or review_count/basic_stats[2]<0.01):
        score=base-0.5
    else:
        score=base+0.5*review_count/basic_stats[2]
    return score


def class_label(row):
    """
    Re-generate Class Labels
    """
    
    if(row['Resturant_Rating_New']<=3.05):
        return 0
    elif(row['Resturant_Rating_New']>=3.6):
        return 2
    else:
        return 1

def data_pre_processing(data):
    """
    Wrapper function for data preprocessing
    """
    print(pd.unique(data['rating']))
    group_count={}
    for i in pd.unique(data['rating']):
        temp_mean=np.mean(data[data['rating']==i]['review_count'].values)
        temp_std=np.std(data[data['rating']==i]['review_count'].values)
        temp_max=np.max(data[data['rating']==i]['review_count'].values)
        group_count[i]=[temp_mean,temp_std,temp_max]
    
    data['Resturant_Rating_New']=data.apply(lambda row: fusion_matrix(row,group_count),axis=1)
    data['class']=data.apply(lambda row: class_label(row),axis=1)
    print(data['class'].value_counts())
    
#    df_majority = data[data['class']!=2]
#    df_minority = data[data['class']==2]
#    df_minority_upsampled = resample(df_minority, 
#                                     replace=True,     # sample with replacement
#                                     n_samples=2000,    # to match majority class
#                                     random_state=123)
#    data= pd.concat([df_majority, df_minority_upsampled])
    return data 

def addModels():
    """
    Classfier Lists to Run
    """
    #initialize a set of models to train 
    models = []
    models.append(('Logistic Regression', LogisticRegression(random_state=0, solver='lbfgs',
                                                            multi_class='multinomial')))
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 20)))
    models.append(('DecisionTree', DecisionTreeClassifier()))
    models.append(('NaiveBayes', GaussianNB()))
    models.append(('SVM', svm.SVC(kernel='rbf',probability = True,gamma=
                                 'scale')))
    models.append(('RandomForest', RandomForestClassifier(n_estimators=500,
                            max_depth=15, min_samples_leaf=2, 
                           min_samples_split=3)))
    return models

def classifier_run(X,Y, model):
    """
    Cross Validation, Plotting of ROC and HeatMap, Printing Confusion Matrix
    """
    
    for item in model: 
        print("====================================================")
        
        y_bin = label_binarize(Y, classes=[0, 1, 2])
        n_classes = y_bin.shape[1]
        model_name=item[0]
        
        print("Running "+ model_name )
        pipe= Pipeline([('clf', item[1])])
        y_score = cross_val_predict(pipe, X, Y, cv=10 ,method='predict_proba')
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(3), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for multi-class data using '+ model_name)
        plt.legend(loc="lower right")
        plt.savefig('Predictive_Analytics/ ROC Plot Using '+ model_name+'.jpg')
        #plt.show()
        plt.close()
        test_size = 0.2
        seed = 7
        X_train, X_validate, Y_train, Y_validate = train_test_split(X, \
                                        Y, test_size=test_size, random_state=seed)
        clf=item[1]
        clf.fit(X_train,Y_train)
        predictions=clf.predict(X_validate)
        
        print(accuracy_score(Y_validate, predictions))
        print(confusion_matrix(Y_validate, predictions))
        print(classification_report(Y_validate, predictions))
        
        conf_mat=confusion_matrix(Y_validate, predictions)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu",\
                    xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.title('Cofusion Matrix Heatmap using '+ model_name)
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.setp(ax.get_yticklabels(), rotation=45)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('Predictive_Analytics/ Cofusion Matrix Heatmap using '+ model_name+'.jpg')
        #plt.show()
        plt.close()
        print("====================================================")


def main():
    """
    Main Function for data preprocesing, normalization and upsampling
    """
    data=pd.read_csv('RawData/Full_Information_Cleaned.csv',index_col=0)
    data=data_pre_processing(data)
    
    X=data[['Accept_Credit_Card', 
           'Outdoor_Seating', 'Take_out', 'Takes_Reservations', 'WIFI',
            'Noise_Level', 'atm', 'bank', 'bar',
           'beauty_salon', 'bus_station', 'cafe', 'gym', 'school',
           'White population', 'Black population', 'American Indian population',
           'Asian population', 'Hispanic or Latino population',
           'High school or higher', 'Graduate or professional degree',
           'Unemployed', 'average_price']]
    Y=pd.factorize(data['class'])[0]
    # To Normalize the feature data into same scale
    norm=Normalizer()
    X=norm.fit_transform(X)
    # To Standarize the data to mean 0 and std 1 
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    
    # Upsampling to deal with Imbalced Class 
    sm = SMOTE(random_state=42)
    X,Y=sm.fit_resample(X, Y) 
    print('Resampled dataset shape %s' % Counter(Y))
    # Binarize the output
    y_bin = label_binarize(Y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    model=addModels()
    classifier_run(X,Y, model)

if __name__ == "__main__":
	 main()
