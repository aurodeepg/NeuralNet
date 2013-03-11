--------------------------------------------------------------------------------------------------------------
README
--------------------------------------------------------------------------------------------------------------
Project files:
1. train.py - Generates 5 output files with weights of the trained network after epochs 0,10,100,1000,10000

2. execute.py - Performs classification and displays the confusion matrix. 
   The profit loss output based on the classification and a csv file called classified_output.csv

USAGE GUIDANCE
==================================================================

train.py
__________________________________________________________________
('Usage: python2.7 train.py CSVFilename')
__________________________________________________________________

CSVFilename - train_data.csv
__________________________________________________________________



execute.py 
__________________________________________________________________
('Usage: python2.7 execute.py TrainingFilename CSVFilename')
__________________________________________________________________

TrainingFilename - training_run_zero.csv,
		   training_run_ten.csv,
		   training_run_hundred.csv,
		   training_run_thousand.csv,
        	   training_run_ten_thousand.csv

CSVFilename - train_data.csv/test_data.csv
__________________________________________________________________
