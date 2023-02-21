PROJECT A
-----------------------------------------------------------
Create the DB, SCHEMA, POPULATE SCHEMA, EXPORT DATA TO CSV:

1. Go to folder: part_a/scripts_sql
2. in _go.bat change psqlPath if necessary
3. run _go00.bat to createDB
4. run _go01.bat to create SCHEMA
5. run _go02.bat to populate SCHEMA
6. run _go03.bat to EXPORT data to CSV
------------------------------------------------------------
RUN CLASSIFIER MODELS:

1. Go to folder:  part_a/part_a_python

2. To use 1R classifier run ex_7_1r_class.py. The class
used to create the 1R model is called ModelOneR.

3. To use ID3 and NaiveBayesClassifier run ex_8_class_bayes.py.
The class used to create this classifiers is called DeployableModel.
You have to choose which type of classifier to use to create the
model when creating the DeployableModel object, passing as paramether: "GaussianNB" or "CategoricalNB" or"ID3".

4. The file extra_createCsvFileClassified.py uses a csv or tab file
with a dataset where attributes are unclassified (example file: d01_unclass.csv) and creates a new csv file 
with the attributes classified using the models created in the DeployableModel class.

--------------------------------------------------------------------
PROJECT A1

CONVERT CSV FILE TO TAB AND CLASSIFY THE DATASET WITH 1R CLASSIFIER
--------------------------------------------------------------------
1. Go to folder: part_a_1/python

2. Running ex_2_convert_csv_to_tab.py, you can convert a csv file to a tab file in the right format.
The original file is called (dataset_long_name_ORIGINAL.csv) and the new created file is
dataset_long_name_CONVERTED.tab.

3. Running ex_3_save_file_output.py the tab file is classified using the 1R Classification Class mentioned previously.
It outputs a txt file called (oneR_OUTPUT.txt) with the output of the 1R classification. You can also test the classifier
with evaluation score, dividing the dataset into train and test and then performing classification.

----------------------------------------------------------------------
ORANGE

1. Go to folder: part_a_1/orange

2. run ex_4.ows

