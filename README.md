--------------------------------------------- **Machine Learning Algorithms** ---------------------------------------------

Description of each directory: 

**Classification:** Scripts to classify different events. These are used to discriminate atm muons(Bkgd) and neutrinos(signal) in KM3NeT/ARCA6 experiment. Project includes: anal_scripts_ROOT_CSV_py/createCSVfromROOT.py to convert ROOT file to csv. 
Different classification algorithms can be trained and tested simultaneously with: BDTClassification_AtmMu_Neu_All.py to find which ones give the best performance. 
For this problem the XGBClassifier from xgboost package was selected. It was trained using: BDTClassification_AtmMu_Neu.py and weights are stored in .sav. Predictions are made using: BDTClassification_AtmMu_Neu_pred.py and are stored in a .csv. 

Use optimising_parameters.py to optimise algorithms parameters and plot_confusion_matrix.py to plot the normalised confusion matrix.   

