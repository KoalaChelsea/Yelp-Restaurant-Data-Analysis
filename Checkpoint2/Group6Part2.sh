#!/bin/bash
cd "$(dirname "$0")" 
pwd
echo '====================================='
echo 'Running Data Clearning'
python DataCleaning/DataClearningAndFeatureEngineering.py
echo '====================================='
echo 'Exploratory Data Analysis'
python Histogram_Correlation/Histogram_Correlation.py
echo 'Association Rule Mining'
echo '====================================='
python AssocationRule/Association_Rule.py
echo 'Clustering Analysis'
echo '====================================='
python Clustering/Clustering.py
echo 'Hypothesis Testing'
echo '====================================='
python Hypothesis\ Testing/hypothesis\ test.py
echo 'Predictive Analytics'
echo '====================================='
python Predictive_Analytics/PredictiveAnalytcis.py
