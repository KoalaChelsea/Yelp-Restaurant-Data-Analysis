# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 02:04:32 2018

@author: KoalaChelsea
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%

def main():
    ## Read this data using pandas as a dataframe.
    data = pd.read_csv("RawData/Full_Information_Cleaned.csv")
    print(data.columns)
    
    # Remove some extreme value in review count
    def manipulate_review_count(data):
        data_index=0
        for i in data['review_count']:        
            if i>=3000 or i<=10:
                data=data.drop(data_index)
            data_index=data_index+1
        return data
    data = manipulate_review_count(data)
    
    # Grouped barplot
    sns.catplot(x="rating", y="review_count", hue="average_price", data=data,
                    height=6, kind="bar", palette="muted")
    sns.despine(left=True)
    plt.title('Grouped barplots')
    plt.savefig('Histogram_Correlation/Grouped_barplots.jpg')
    plt.show()
    plt.close()
    
    
    # Make a list of continuous variables
    continuous_list = ['review_count','atm','bank','bar','beauty_salon','bus_station','cafe',
                       'gym','school','White population','Black population',
                       'American Indian population','Asian population',
                       'Hispanic or Latino population','High school or higher',
                       'Graduate or professional degree','Unemployed']
    dataContinuous = data[continuous_list]
    
    # Compute the correlation matrix
    corr_con = dataContinuous.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_con, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_con, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt='.2f')
    f.subplots_adjust(top=0.93)
    plt.title('Correlation Matrix for All Continuous Variables')
    plt.savefig('Histogram_Correlation/Correlations_Variables.jpg')
    plt.show()
    plt.close()
    
    # select three interested variables
    selected_list = ['cafe', 'bar', 'Unemployed']
    dataSelected = data[selected_list]
    
    # Compute the correlation matrix
    corr_sec = dataSelected.corr()
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 4, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_sec, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt='.2f')
    f.subplots_adjust(top=0.93)
    plt.title('Correlation Matrix for Selected Continuous Variables')
    plt.savefig('Histogram_Correlation/Correlations_Categorical_Variables.jpg')
    plt.show()
    plt.close()
    
    # Pairplot for selected data
    g = sns.pairplot(dataSelected)
    plt.title('Pairwise Plots for Selected Continuous Variables')
    plt.savefig('Histogram_Correlation/PairWisePlot.jpg')
    plt.show()
    plt.close()
    
    
    # Histogram 1
    fig, ax = plt.subplots(figsize=(7,7),sharey=True)
    sns.distplot(data["atm"] , color="skyblue", label="atm")
    sns.distplot(data["bank"] , color="pink", label="bank")
    ax.legend()
    plt.title('Histograms of atm and bank')
    plt.savefig('Histogram_Correlation/Histogram1.jpg')
    plt.show()
    plt.close()
    
    
    # Histogram 2
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    sns.distplot(data['White population'], color="steelblue", ax=axes[0, 0])
    sns.distplot(data['Black population'], color="olive", ax=axes[0, 1])
    sns.distplot(data['Asian population'], color="gold", ax=axes[1, 0])
    sns.distplot(data['Hispanic or Latino population'], color="teal", ax=axes[1, 1])
    plt.savefig('Histogram_Correlation/Histogram2.jpg')
    plt.show()
    
    plt.close()

if __name__ == "__main__":
	 main()
