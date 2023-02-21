# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:58:32 2021

@author: duart
"""

import numpy as np
import sys
import Orange as DM
from Orange.data import Instance,Table
from ex_7_1r_class import ModelOneR


# write the output of ONER to a text file
def writeOutput(fileName, model):

    file1 = open(fileName,"w")

    for i in range ( len(model)):
    
        strAtrVal = "( "+str(model[i][0]) +", "+str(model[i][1])+", "+ str(model[i][2])+ " )"+" : ("+str(model[i][4])+", "+str(model[i][5])+") \n"
        file1.write(strAtrVal)
    file1.close()



fileName = "dataset_long_name_CONVERTED.tab"

print("-------------------------- write to txt ---------------------------------------")


oneR = ModelOneR(fileName, 0)
model = oneR.fitOneR()
print(model)

fileOutName="oneR_OUTPUT.txt"
writeOutput(fileOutName,model)



print("-------------------------- evaluate ---------------------------------------")
'''
oneR = ModelOneR(fileName, 0.3)
model = oneR.fitOneR()

trainDataSet = oneR.getTrainDataSet()
testDataSet = oneR.getTestDataSet()
   
#calculate the labels and the predicted values for the text dataset
#in order to evaluate the results
target = oneR.getLabels(testDataSet)
pred = oneR.predict(model, testDataSet)

print(model)


#evaluate the model 
oneR.evaluate(target,pred)
'''



