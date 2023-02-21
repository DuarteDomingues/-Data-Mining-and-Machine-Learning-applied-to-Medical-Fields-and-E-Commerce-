# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:40:38 2021

@author: duart
"""

import csv
import os
from csv import reader



'''
runs through the csv file, and returns an array with all the rows, with
the three headed row formated included.
'''
def createArrayWithThreeHeadedRow(fileName):

    rowDiscrete = ['discrete']*23
    rowClasses = ['class'] + ['']*22

    newRows = []

    # open file in read mode
    with open(fileName, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        c=0
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            if (c ==1):
                newRows.append(rowDiscrete)
                newRows.append(rowClasses)
            
            c=c+1
            newRows.append(row)
    return newRows



'''
receives an array of rows, and a name for the output file, and writes all the rows into a tab file
with the especific tab format.
'''

def writeToTab(rows, newFile):

    for line in rows:
       with open(newFile, 'a',  newline="") as new_txt:    #new file has .tab extn
           txt_writer = csv.writer(new_txt, delimiter = '\t') #writefile
           txt_writer.writerow(line)   #write the lines to file`


#Original CSV fileName
fileName = "dataset_long_name_ORIGINAL.csv"
#Output TAB fileName
fileOutput = "dataset_long_name_CONVERTED.tab"
#create a list with the data in the three row header format
rows = createArrayWithThreeHeadedRow(fileName)
rows = rows[:-1]
#write the rows to a tab file
writeToTab(rows, fileOutput)


