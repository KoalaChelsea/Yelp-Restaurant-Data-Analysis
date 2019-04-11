#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 23:18:27 2018

@author: james
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import r2_score


#%%

def Anova_Test_For_Price_Review_count(myData):
    mod = ols('review_count ~ price',data=myData).fit()
    print(mod.summary())
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(aov_table)
    res = mod.resid 
    sm.qqplot(res, line='s')
    plt.title('Q-Q Plot for Anova Test')
    plt.savefig('Hypothesis Testing/QQplot.jpg')
    plt.show()
    plt.close()
    ## Verify using pair-wise T-Test
    print("\n\n")
    print("T-Test of price group '$' against '$$','$$$', '$$$$'")
    ttest1 = stats.ttest_ind(myData['review_count'][myData['price'] == '$'], 
                             myData['review_count'][myData['price'] == '$$'])
    print(ttest1)
    ttest2 = stats.ttest_ind(myData['review_count'][myData['price'] == '$'], 
                             myData['review_count'][myData['price'] == '$$$'])
    print(ttest2)
    ttest3 = stats.ttest_ind(myData['review_count'][myData['price'] == '$'],
                             myData['review_count'][myData['price'] == '$$$$'])
    print(ttest3)


def Ratings_Review_count(myData):
    data_x = myData['rating']
    data_y = myData["review_count"]
    slope, intercept, r_value, p_value, std_err = stats.linregress(data_x, data_y)
    print('R-value for this fiting is '+str(r_value))
    print('p-value for this fiting is '+str(p_value))
    plt.figure(figsize=(10,6))
    plt.plot(data_x, data_y, 'o', label='original data')
    plt.plot(data_x, intercept + slope*data_x, 'r', label='fitted line')
    plt.xlabel('ratings')
    plt.ylabel('review_counts')
    plt.legend()
    plt.title('Ratings V.S Review _count')
    plt.savefig("Hypothesis Testing/regression.png")
    plt.show()


def main():
    myData = pd.read_csv('RawData/Full_Information_Cleaned.csv', sep = ',',encoding = 'utf-8-sig')
    Anova_Test_For_Price_Review_count(myData)
    Ratings_Review_count(myData)

if __name__ == "__main__":
	 main()





