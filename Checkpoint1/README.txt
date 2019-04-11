There are 1 file and 7 subfolders in this checkpoint folder.


==========================================================================
Please note that there is a sample code to test-out in Phase1Testing Folder.
The data extraction from API and Scrape are sample-based there.
Otherwise the code takes over 1 day to finish the whole data extraction process.  
Other folders all contains actual scripts and data. 
*******How to Run********
There is only one python file within Phase1Testing Folder. 
You can just trigger the py file from your command line using following command:  

python Phase1Testing/TestingSample.py

Please make sure that your current directory is at Project1Submission. 

Please note that we require 'yelpapi' package to run this code.
==========================================================================

Following describe each folder and structure:
==========================================================================
1.DataPreProcessing: Folder for data pre-processing 

Column_Definition.csv: Column Definition and Min/Max Values of applicable columns. 
                       This is used for data cleanness check. 
DataCleannessCheck.py: Script to run the health check.

DataMerger.py:         Script to merge the data, since the data extraction is city based.

Health_report.py:      Screen output for Data Cleanness check.

NewFeature.py:         Script to generate and print out three new features. 
==========================================================================
2. DataPulling: Folder for Yelp API Data Extraction 

PullingRawData.py:  python code to extract data using Yelp API 

***_neighborhood.txt: Lists of neighbourhood in different cities. 
==========================================================================
3. DataScrape:

WebScrape.py:           WebScrape Script to retrieve data from Yelp Website. 
==========================================================================
4. GoogleMapAPI: Get Data from Google Maps Places API 

google_map_api.py:      Retrieve Data from Google Maps API

GoogleAPIDataTransformation.py: Raw Google Maps Data Transformation. 
==========================================================================
5. Phase1Tesing: 

TestingSample.py: Sample code to test functionality for each section 

TestingSampleScreenShot.txt: std out sample from the python file above. 
==========================================================================
6. RawData: 

Resturant_API_Combined.csv: Raw data from Yelp API

Resturant_Map_API_Combined.csv: Raw data from Google Maps API after Transformation.

Resturant_Scrape_Combined.csv: Raw Data from Yelp Web Scrape

Resturant_Full_Infomation.csv: Raw data after joining the three data sets above. 
==========================================================================
7. Demography: 

demography.py: Web scrape data from city-data.com

Input.txt: A series of txt data including zip code of restuarants.

data.csv: A series of csv data of information scrapped from city-data.com
==========================================================================









 
