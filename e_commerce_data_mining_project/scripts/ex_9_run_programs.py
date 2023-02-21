# -*- coding: utf-8 -*-


from subprocess import Popen
from _goPy_transform_v02 import generateBasket,generateDataFile_basket,generateDataFile_tab

#open bat file
p = Popen("_go05.bat")
stdout, stderr = p.communicate()

poll = p.poll()

#check is subprocess is alive
alive=True
while alive:
    if poll is not None:
        alive=False


#run python file, convert z_dataset_sample_OUT.TXT to a basket and tab file

fIN  = "z_dataset_sample_OUT.txt" #"z_abstract_test.txt" #"z_dataset_sample_OUT.txt"
fOUT = "zz_dataset_2012_01" #"z_abstract_test"     #"zz_dataset_2012_01"
print()
print( ">> 1. Generate Basket structure from CSV file: " + fIN )
( basket, all_itemset ) = generateBasket( fIN )

fOUT_basket = fOUT + ".basket"
fOUT_tab    = fOUT + ".tab"
        
print( ">> 2. Generate .basket dataset file: " + fOUT_basket )
generateDataFile_basket( basket, fOUT_basket )
    
print( ">> 3. Generate .tab dataset file: " + fOUT_tab )
generateDataFile_tab( basket, all_itemset, fOUT_tab )
            
