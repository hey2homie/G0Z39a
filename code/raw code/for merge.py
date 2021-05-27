# -*- coding: utf-8 -*-

"""


@author: mianyong
"""
#function, to merge the excel and csv in the same folder.At least one csv file should be exsit in the folder


import os
import pandas as pd

cwd = os.path.abspath('') 
files = os.listdir(cwd)  

#to get all the names of excel in this folder
excellist=[]
for file in files:
    if file.endswith('.xlsx'):
       excellist.append(file)


#to find all csv files in this folder
csvlist=[]
for file in files:
    if file.endswith('.csv'):
       csvlist.append(file)
       
       
ace=pd.read_csv(csvlist[0])   
          
num2=len(csvlist)
#merge the csv files with the merged excel
for x in range(1,num2):
    ace=ace.merge(pd.read_csv(csvlist[x]),on="country",how="outer")     

#The first excel in the folder
if excellist[0] is not None:
        num=len(excellist)
        for x in range(0,num):
            ace=ace.merge(pd.read_excel(excellist[x]),on="country",how="outer")


#change the nmae according to year
ace.to_csv('wholedata.csv')