# to use accented characters in the code
# -*- coding: cp1252 -*-

import sys
import Orange as DM
import numpy as np
from sklearn.model_selection import train_test_split
from Orange.data import Instance,Table
import math
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, classification_report
import pickle


class ModelOneR():
    
    def __init__(self,fileName, testsize=0.3):
        
        self.__dataset= Table(fileName)
        print(self.__dataset)
        self.__trainDataSet, self.__testDataSet = self.__splitDataset(self.__dataset,testsize)
    
    # split the dataset based on a test size            
    
    def getTrainDataSet(self):
        
        return self.__trainDataSet
    
    def getTestDataSet(self):
        
        return self.__testDataSet
    
    
    def __splitDataset(self, dataset, testsize):
        
        #shuffle the dataset
        dataset.shuffle()
    
        trainSize = int(len(dataset)*(1-testsize))
        trainDataSet = dataset[0:trainSize]
        testDataSet = dataset[trainSize:len(dataset)]
    
        return trainDataSet,testDataSet
    
    #aux method to return a list with the attributes names
    def __getAttributesList(self, dataset):
    
        attributesList=[]
        #datasetDomain = dataset
        for i in  range(len(dataset.domain)): 
            if ( dataset.domain[i] !=dataset.domain.class_var):
                attributesList.append(str(dataset.domain[i]))
        return attributesList
    
    
    # get the variable dataset-structure given a string with its name
    def __get_variableFrom_str(self, dataset, str_name ):
        variable_list = dataset.domain.variables
        for variable in variable_list:
            if( variable.name == str_name ): return variable
   
        return None 
    
    
    # contingencyMatrix; i.e., the joint-frequency table
    # this implementation does not account for missing-values
    # (i.e., missing-values are not included in the variables-domain)
    # M(row, column)
    def __get_contingencyMatrix(self, dataset, rowVar, colVar ):
        if( isinstance( rowVar, str ) ): rowVar = self.__get_variableFrom_str( dataset, rowVar )
        if( isinstance( colVar, str ) ): colVar = self.__get_variableFrom_str( dataset, colVar )
        if( not (rowVar and colVar) ): return ( [], [], None )
        if( not (rowVar.is_discrete and colVar.is_discrete) ):
      
            return ( [], [], None )
   
        rowDomain, colDomain = rowVar.values, colVar.values
        len_rowDomain, len_colDomain = len( rowDomain ), len( colDomain )
        contingencyMatrix = np.zeros( (len_rowDomain, len_colDomain) )
        for instance in dataset:
            rowValue, colValue = instance[rowVar], instance[colVar]
            if( np.isnan( rowValue ) or np.isnan( colValue ) ): continue
      
            rowIndex, colIndex = rowDomain.index(rowValue), colDomain.index( colValue )
            contingencyMatrix[ rowIndex, colIndex ] += 1
        return ( rowDomain, colDomain, contingencyMatrix )
    
    # P( H | E )
    # H means Hypothesis, E means Evidence
    # a frequency approach
    def __get_conditionalProbability(self, dataset, H, E ):
        if( isinstance( H, str ) ): H = self.__get_variableFrom_str( dataset, H )
        if( isinstance( E, str ) ): E = self.__get_variableFrom_str( dataset, E )
        if( not (H and E) ): return ( [], [], None )
        ( rowDomain, colDomain, cMatrix ) = self.__get_contingencyMatrix( dataset, H, E )

        len_rowDomain, len_colDomain = len( rowDomain ), len( colDomain )
        E_marginal = np.zeros( len_colDomain )
        for col in range(len_colDomain): E_marginal[col] = sum( cMatrix[:, col] )
   
        for row in range(len_rowDomain):
            for col in range(len_colDomain):
                cMatrix[row, col] = cMatrix[row, col] / E_marginal[col]
        return ( rowDomain, colDomain, cMatrix )
    
    # 1R Related Functions_______________________________________________________________________________
    # error matrix for a given feature and considering the datatset class
    def __get_errorMatrix( self,dataset, feature ):
        if( isinstance( feature, str ) ): feature = self.__get_variableFrom_str( dataset, feature )
        the_class = dataset.domain.class_var
        ( rowDomain, colDomain, cMatrix ) = self.__get_conditionalProbability( dataset, the_class, feature )
        if( not (rowDomain or colDomain) ): return ( [], [], None )

        errorMatrix = 1 - cMatrix
        return ( rowDomain, colDomain, errorMatrix )
    
    
    #_______________________________________________________________________________
    # P( H | E )
    # show matriz and textual description
    def __show_conditionalProbability( self, dataset, H, E ):
        ( rowDomain, colDomain, cMatrix ) = self.__get_conditionalProbability( dataset, H, E )
        print( cMatrix )
        print()

        for h in rowDomain:
            for e in colDomain:
                rowIndex, colIndex = rowDomain.index( h ), colDomain.index( e )
                P_h_e = cMatrix[ rowIndex, colIndex ]
                print( "  P({} | {}) = {:.3f}".format( h, e, P_h_e ) )
    
    # aux function to count the number of times a value of an attribute occurs
    def __getValueFreq(self, dataset, attrib, value):
    
        myope_subset = [d for d in dataset if d[attrib] == value]
        return len(myope_subset)

    
    #implementation of 1R classifier
    def fitOneR(self):
        # i have attribute, and value names
        #dic (errorByAtrrib, list(feature,value)
        attributes_list = self.__getAttributesList(self.__trainDataSet)
        
        dicOneR={}
        errorTotalsDic={}
        for atr in attributes_list:
            
            ( classDomain, featureDomain, errorMatrix ) = self.__get_errorMatrix( self.__trainDataSet, atr )
            if( not (classDomain or featureDomain) ): return
            totalErr=0
            listPairs = []
            for feature in range(len(featureDomain)):
            
                errorFeature = errorMatrix[:, feature]
                errorMin = min( errorFeature )
   
                if (math.isnan(errorMin)==False):
                    errorMinIndex = errorFeature.tolist().index( errorMin )
                    featureValue = featureDomain[feature]
                    classValue = classDomain[errorMinIndex]
                    showStr = "(" + atr + ", " + featureValue + ", " + classValue + ") : "
                    print( showStr + "{:.3f}".format( errorMin ) )
                    freqVal = self.__getValueFreq(self.__trainDataSet,atr,featureValue)
                    numError = (int)(freqVal*errorMin)
                    err = freqVal * (1-errorMin)
                    totalErr=totalErr+err
                    listPairs.append([atr,featureValue,classValue,errorMin,numError])
                    if (feature == len(featureDomain)-1):
                        totalErr = 1- totalErr / len(self.__trainDataSet)
                        print("error/ total: ", totalErr, "\n")
                        errorTotalsDic[atr] = totalErr
                        totalErr=0
                        dicOneR[atr] = listPairs
    
        #choose the attribute with minimal total Error
        bestAtr = min(errorTotalsDic, key=errorTotalsDic.get)
        #return the pairs (feature,value) of attribute with min error
        print("One-R \n")
        dic = self.__getPairOccurence(dicOneR[bestAtr])
        return (dic)
        

    def __getPairOccurence(self, dic):

        for i in range (len(self.__trainDataSet)):
            for feat in self.__trainDataSet.domain.attributes:
                if str(feat) == str(dic[0][0]):
                    for j in range(len(dic)) :
                        if (dic[j][1] == self.__trainDataSet[i][feat]):
                            if (dic[j][2] ==self.__trainDataSet[i][self.__trainDataSet.domain.class_var]):
                    
                               if (len(dic[j])>5):
                                   dic[j][5] = dic[j][5] +1
                                   
                               else:
                                   dic[j] = dic[j] + [1]
        return dic
    
    # predict the labels for a dataset based on a OneR trained model
    def predict(self, dic, dataset):
    
        pred=[]
        for i in range (len(dataset)):
        
            for feat in dataset.domain.attributes:
                if str(feat) == str(dic[0][0]):

                    for j in range(len(dic)) :
                        if (str(dataset[i][feat]) ==dic[j][1]):
                            pred.append(dic[j][2])
        return pred
    
    # aux function to get the class values from a dataset
    def getLabels(self, dataset):
    
        target = []
        for i in range (len(dataset)):
            #add class to target
            target.append(str(dataset[i][dataset.domain.class_var]))
        
        return target
    
    
    
    # evaluate performance matrics of a dataset based on the target and the predicted class values
    def evaluate( self, target, targetPred):
    
        target = np.asarray(target)
        targetPred = np.asarray(targetPred)
        count_misclassified = (target != targetPred).sum()
        print("OneR")
        print("=" * 30)
        print('Misclassified samples: {}'.format(count_misclassified))
        accuracy = accuracy_score(target, targetPred)
        print('Accuracy: {:.2f}'.format(accuracy))
    
        print("Recall score : ", recall_score(target, targetPred , average='weighted'))
        print("Precision score : ",precision_score(target, targetPred , average='weighted'))
        print("F1 score : ",f1_score(target, targetPred , average='weighted'))
    
        print(classification_report(target, targetPred))
        print("Confusion Matrix:")
        print(confusion_matrix(target, targetPred))
        
    
    #save oneR dictionary of the trained dataset to a dic
    def saveModel(self, model):
    
      modelPickle={}
      modelPickle['model1R'] = model
      

      with open('modelPickleOneR.pickle', 'wb') as handle:
          pickle.dump(modelPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


if __name__=="__main__":
   
   fileName = "./dataset_lenses.csv"
   

   #Create Model OneRClassifier with 0.3 percentage of data for test
   oneR = ModelOneR(fileName, 0.3)
   #fit model
   model = oneR.fitOneR()
   print(model)
   
   #get the train and test data set
   trainDataSet = oneR.getTrainDataSet()
   testDataSet = oneR.getTestDataSet()
   
   #calculate the labels and the predicted values for the text dataset
   #in order to evaluate the results
   target = oneR.getLabels(testDataSet)
   pred = oneR.predict(model, testDataSet)
       
   #evaluate the model 
   oneR.evaluate(target,pred)
       
   #save oneR Classification model (dic) to a pickle file
   oneR.saveModel(model)
   
   
   
   
   
   
   
