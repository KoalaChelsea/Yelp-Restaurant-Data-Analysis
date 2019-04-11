# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 23:43:31 2018

@author: KoalaChelsea
"""
mport urllib.request
import pandas as pd
import time

def User_interface():
    '''
    Function Name: User_interface()
    Input: None
    Output: zip_code with list type
    
    Allow user enter one zipcode to manipulate the population analysis. 
    Shut down the user interface if you want to read zipcode from a file, 
    an go to Open_file() to turn it in
    '''
    
    zip_code=input('Please enter a zipcode: ')
    return list(zip_code)

def URL_determine(zip_code):
    '''
    Function Name: URL_determine()
    Input: one zip_code
    Output: the URL link with the zipcode
    
    Find URL of the zipcode, and return the URL address
    '''
    
    Base_URL='http://www.city-data.com/zips/'
    URL=Base_URL+zip_code+'.html'
    return URL

def Get_data(URL):
    '''
    Function Name: Get_data()
    Input: URL
    Output: The string-type html code of the URL
    
    Get all html code of the URL in text to return
    '''
    
    data=urllib.request.urlopen(URL).read()
    #z_data=data.decode('UTF-8')
    return str(data)

def Get_population(data):
    '''
    Function Name: Get_population()
    Input: data
    Output: Get the total population and return with int type
    Get total population in 2010    
    '''
    if '<b>Estimated zip code population in 2016:</b>' in data:
        target_str='<b>Estimated zip code population in 2016:</b>'
    else:
        target_str='<b>Estimated zip code population in 2015:</b>'
    population=''
    pos=data.find(target_str)
    start=pos+len(target_str)
    line=data[start:start+100]
    for i in line:
        if i=='<':
            break
        elif i.isdigit():
            population=population+i
            continue
        else:
            continue
    print('Total Population:', population)

    try:
        population=int(population)
        return population
    except:
        return 0

def Get_race(data, race_name):
    '''
    Function Name: Get_race()
    Input: data, race_name
    Output: a list of population of each race
    
    Get the population of each race.
    '''
    
    ls=[]
    for i in race_name:
        population=''
        target_str='</span>'+i+'</li>'
        end=data.find(target_str)
        race_line=data[end-100: end]
        for i in reversed(race_line):
            if i=='>':
                break
            elif i.isdigit():
                population=population+i
                continue
            else:
                continue
        ls.append(int(population[::-1]))
        
    return ls


def Get_rate(data, education):
    '''
    Function Name: Get_rate()
    Input: data, education
    Output: Get necessary data to return
    
    Get necessary data to return
    '''
    
    ls=[]
    for i in education:
        rate=''
        target_str='<li><b>'+i+':</b>'
        pos=data.find(target_str)
        start=pos+len(target_str)
        line=data[start:start+100]
        for j in line:
            if j=='<':
                break
            elif j.isdigit() or j=='%' or j=='.':
                rate=rate+j
                continue
            else:
                continue
        if i=='Mean travel time to work (commute)':
            rate=rate+' minutes'
        ls.append(rate)
    return ls

def Data_collect(ls):
    '''
    Function Name: Data_collect()
    Input: ls
    Output: A correct list to return
    
    Manipulate each element in the list, take same-index element from each list to construct a new list,
    and append these lists to a new list, and return the outtest list.
    '''
    
    ls2=[]
    for j in range(len(ls[1])):
        ls3=[]
        for i in ls:
            ls3.append(i[j])
        ls2.append(ls3)
    return ls2

def open_file():
    '''
    Function Name: open_file()
    Input: None
    Output: zipcode list which is read from the file
    
    *** If you want to read zipcode from a file, please shut down the user interface
    *** For each file type, you can just use the corresponding library to open the file,
        and do a data clean to get the zipc0de c0lumn, and cast any type of zipcode to a list
    *** You must return a list of zipcode whatever any fiels to read
    '''
    
    ls=[]
    with open('input9.txt', 'r') as f:
        while True:
            line=f.readline()
            if line=='':
                break
            ls.append(line.strip())
    return ls


def main():
    '''
    Function Name: main()
    Input: None
    Output: None
    
    This is main function to control the whole data analysis of population with the zipcode
    '''
    
    #zip list
    zip_ls=open_file()
    
    #title lists
    
    race_name=['White population', 'Black population', 'American Indian population', \
               'Asian population', 'Native Hawaiian and Other Pacific Islander population', \
               'Some other race population', 'Two or more races population', \
               'Hispanic or Latino population']
    education=['High school or higher', 'Bachelor\'s degree or higher', \
               'Graduate or professional degree', \
               'Unemployed', 'Mean travel time to work (commute)']
    married_status=['Never married', 'Now married', 'Separated', 'Widowed', 'Divorced']    
    #value lists
    population_ls=[]
    race_ls=[]
    education_ls=[]
    married_ls=[]

    new_zipcode_ls=[]
    error_zipcode_ls=[]
    
    #Write zip code to csv file
    csv_dict={'Zipcode': zip_ls}
    
    #Get total population
    for zip_i in zip_ls:
        try:
            print('zip_i:', zip_i)
            URL=URL_determine(zip_i)
            data=Get_data(URL)   

            race_pop=Get_race(data, race_name)
            race_ls.append(race_pop)

            education_rate=Get_rate(data, education)
            education_ls.append(education_rate)

            married_rate=Get_rate(data, married_status)
            married_ls.append(married_rate)
            new_zipcode_ls.append(zip_i)

            population=Get_population(data)
            population_ls.append(population)
            time.sleep(1)
        except:
            print('ERROR ZIPCODE', zip_i)
            error_zipcode_ls.append(zip_i)
            continue

    csv_dict['Zipcode']=new_zipcode_ls
    csv_dict['Zip code population in 2016']=population_ls

    #Get Race Population
    race_ls=Data_collect(race_ls)
    for i in range(len(race_name)):
        csv_dict[race_name[i]]=race_ls[i]
            
    #Get Educational Rate and Employed Rate
    education_ls=Data_collect(education_ls)
        
    for i in range(len(education)):
        csv_dict[education[i]]=education_ls[i]
    
    #Get Marriage Rate
    married_ls=Data_collect(married_ls)
    
    
    for i in range(len(married_status)):
        csv_dict[married_status[i]]=married_ls[i]
    
    '''
    Print Error Zipcode
    '''
    print('ERROR ZIPCODES ARE: ')
    for zip_i in error_zipcode_ls:
        print(zip_i)
    print()

    '''
    Test the length of each column
    '''
    test_dict_ls=['Zipcode', 'Zip code population in 2016', \
		'White population', 'Black population', 'American Indian population', \
               'Asian population', 'Native Hawaiian and Other Pacific Islander population', \
               'Some other race population', 'Two or more races population', \
               'Hispanic or Latino population', 'High school or higher', \
               'Bachelor\'s degree or higher', 'Graduate or professional degree', \
               'Unemployed', 'Mean travel time to work (commute)', \
               'Never married', 'Now married', 'Separated', 'Widowed', 'Divorced']
    
    '''
    Print the length of each column
    '''
    for i in test_dict_ls:
        print('The length of column -'+i+'- is', len(csv_dict[i]))
        
    #Construct DataFrame
    dataframe=pd.DataFrame(csv_dict)
    
    dataframe=dataframe[['Zipcode', 'Zip code population in 2016', \
		'White population', 'Black population', 'American Indian population', \
               'Asian population', 'Native Hawaiian and Other Pacific Islander population', \
               'Some other race population', 'Two or more races population', \
               'Hispanic or Latino population', 'High school or higher', \
               'Bachelor\'s degree or higher', 'Graduate or professional degree', \
               'Unemployed', 'Mean travel time to work (commute)', \
               'Never married', 'Now married', 'Separated', 'Widowed', 'Divorced']]
    
    #Write to CSV file
    dataframe.to_csv("data.csv", index=True, sep=',')
    

if __name__ == "__main__":
    main() 