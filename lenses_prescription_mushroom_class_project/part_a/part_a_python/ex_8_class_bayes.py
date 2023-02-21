# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 23:33:40 2021

@author: duart
"""

import numpy as np
import Orange as DM
import pickle
from Orange.data import Instance,Table
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, classification_report
from sklearn import tree



class DeployableModel():
    
    #fileName -> name of the file with the database
    #modelType -> string defines model type ("GaussianNB","CategoricalNB","ID3")
    def __init__(self,fileName, modelType, testsize=0.3,randomstate=7):
        
        self.__dataset= Table( fileName )
        self.__modelType = modelType
        self.__labelEncoder= preprocessing.LabelEncoder()
        self.__ordinalEncoder= preprocessing.OrdinalEncoder()
        self.__testsize = testsize
        self.__randomstate = randomstate
        self.__X_train = []
        self.__X_test = []
        self.__y_train = []
        self.__y_test = []
    
    def __getFeaturesAndTarget(self):
        features=[]
        target=[]
        for i in range (len(self.__dataset)):
            listAtt =[]
        #add class to target
            target.append(str(self.__dataset[i][self.__dataset.domain.class_var]))
        
            for feat in self.__dataset.domain.attributes:
            #add features to list
                listAtt.append(str(self.__dataset[i][feat]))
            features.append(listAtt)
                
        return features,target
                
    def __doEncoding(self):
        
        features,target = self.__getFeaturesAndTarget()
        
        dataEncoded = self.__ordinalEncoder.fit_transform(features)
        label =self.__labelEncoder.fit_transform(target)
        
        return dataEncoded,label
    
    def __doTestSplit(self):
        
        dataEncoded, label = self.__doEncoding()
      #  self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(dataEncoded, label,test_size=self.__testsize, random_state=self.__randomstate)
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(dataEncoded, label,test_size=self.__testsize)

    def fitModel(self):
        
        self.__doTestSplit()
        
        #print(self.__X_train)
       # print(self.__y_train)
        
        model = None
        if (self.__modelType=="GaussianNB"):
            model = GaussianNB()
            model.fit(self.__X_train, self.__y_train)
        
        elif (self.__modelType=="CategoricalNB"):
            model = CategoricalNB()
            model.fit(self.__X_train, self.__y_train)
        
        elif (self.__modelType=="ID3"):
            model = tree.DecisionTreeClassifier(criterion="entropy",max_depth=8,splitter='best')
            model.fit(self.__X_train, self.__y_train)
        return model
    
    def evaluate(self, model):
        

        # PERFORMANCE METRICS 
        y_pred_cnb = model.predict(self.__X_test)
        y_prob_pred_cnb = model.predict_proba(self.__X_test)
        # how did our model perform?
        count_misclassified = (self.__y_test != y_pred_cnb).sum()

        print(self.__modelType)
        print("=" * 30)
        print('Misclassified samples: {}'.format(count_misclassified))
        accuracy = accuracy_score(self.__y_test, y_pred_cnb)
        print('Accuracy: {:.2f}'.format(accuracy))

        print("Recall score : ", recall_score(self.__y_test, y_pred_cnb , average='weighted'))
        print("Precision score : ",precision_score(self.__y_test, y_pred_cnb , average='weighted'))
        print("F1 score : ",f1_score(self.__y_test, y_pred_cnb , average='weighted'))

        print(classification_report(self.__y_test, y_pred_cnb))
        print("Confusion Matrix:")
        print(confusion_matrix(self.__y_test, y_pred_cnb))

        print("\n")
        print("=" * 30)

        for i in range (len(self.__X_test)):
            print("Features values:",self.__ordinalEncoder.inverse_transform(self.__X_test)[i], "\nPredicted class: ",self.__labelEncoder.inverse_transform(y_pred_cnb)[i] )
        
    
    def saveModel(self, model):
        
        modelPickle={}
        modelPickle['model'] = model
        modelPickle['ordinalEncoder'] = self.__ordinalEncoder
        modelPickle['labelEncoder'] = self.__labelEncoder


        with open('modelPickle.pickle', 'wb') as handle:
            pickle.dump(modelPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__=="__main__":

   
    fileName = "./dataset_lenses.csv"
    # Possible Classifiers: "GaussianNB","CategoricalNB","ID3"

    modelDep = DeployableModel(fileName,"CategoricalNB")
    #fit the model
    model = modelDep.fitModel()
    #evaluate the model
    modelDep.evaluate(model)
    #save the model to a picle
    modelDep.saveModel(model)



        
        