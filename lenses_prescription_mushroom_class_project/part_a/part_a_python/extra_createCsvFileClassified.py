# -*- coding: utf-8 -*-
#IMPORTS
import numpy as np
import Orange as DM
import pickle
from Orange.data import Instance,Table
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, classification_report
import pickle
import csv


#load pickle
def loadPickle(fileName):
    
    D = pickle.load( open( fileName, "rb" ) )
    model = D['model']
    ordinalEncoder = D['ordinalEncoder']
    labelEncoder = D['labelEncoder']
    
    return model,ordinalEncoder,labelEncoder


def getFeaturesAndTarget(dataset):

    features=[]

    for i in range (len(dataset)):
        listAtt =[]

        if (str(dataset[i][dataset.domain.class_var])== "?" or str(dataset[i][dataset.domain.class_var])== "" ):
        
            for feat in dataset.domain.attributes:
                #add features to list
                listAtt.append(str(dataset[i][feat]))
            features.append(listAtt)
    
    return features



#WRITING TO A NEW EXCEL FILE
def getTargetsAndFeatures(dataset,model,ordinalEncoder,labelEncoder):
    features =getFeaturesAndTarget(dataset)
   
    for i in range (len(features)):
    
        #encoding feature
        feat2d=[]
        feat2d.append(features[i])
        encodedFeat = ordinalEncoder.transform(feat2d)
        pred = model.predict(encodedFeat)
        predDecoded = labelEncoder.inverse_transform(pred)
        features[i].append(predDecoded[0])
    
    return features



def writeToCSV(features, fileName):
    fields = ["age", "prescription", "astigmatic","tear_rate","lenses"]
    types = ["discrete"]*5
    csvFileName = fileName

    with open(csvFileName, 'w', newline="") as csvfile: 
    
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
    
        # writing the attributes 
        csvwriter.writerow(fields) 
        # writing the type
        csvwriter.writerow(types)
    
        for feat in features:
            csvwriter.writerow(feat)





#READ DATA FROM FILE
fileName = "./d01_unclass.csv"
dataset = Table( fileName )

#load pickle
pickleName="modelPickle.pickle"
model,ordinalEncoder,labelEncoder= loadPickle(pickleName)

#get features that don't have a target from a csv file
features=getFeaturesAndTarget(dataset)

#get features and predicted label in an array
featuresAndTarget=getTargetsAndFeatures(dataset,model,ordinalEncoder,labelEncoder)

#writeToCSV
writeToCSV(featuresAndTarget, "dataset_classified.csv")










    
    
    