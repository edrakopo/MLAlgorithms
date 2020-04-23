import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

#------- Read .csv files -------

predictions = pd.read_csv("predictions/MLP_predictions_single_noDistsVarSkewKurt.csv", header = None)

pred_index = predictions.iloc[:,0]
true = predictions.iloc[:,1]
pred = predictions.iloc[:,2]

print("index:", pred_index)
print("true: ",true)
print("pred: ",pred)

original_indices = list()

file = open("data/X_test_indices.dat",'r')
lines = file.readlines()
for x in lines:
    original_indices.append(int(x))
file.close()

print("original indices: ", original_indices)

#------- Select correct and wrong prediction data sets -------

mislabeled_index = list()

for i in range(len(true)):
    if (true[i]==pred[i]):
        continue    #print("Correct prediction, yay :)")
    else:
        mislabeled_index.append(original_indices[i]) #print("Incorrect prediction :(")

out_file = open("Misclassified_EventNumbers.dat","w")
for index in mislabeled_index:
    out_file.write("%i \n" % index)
out_file.close()

#-------- Convert misclassified index numbers to misclassified event numbers --------

muon_evnum = pd.read_csv("data/pdf_muon_Parametric_single_wEvNum.csv", header = None)
evnum_muons = muon_evnum.iloc[:,11]

electron_evnum = pd.read_csv("data/pdf_electron_Parametric_single_wEvNum.csv", header = None)
evnum_electrons = electron_evnum.iloc[:,11]

print("evnum muons: ", evnum_muons)
print("evnum electrons: ",evnum_electrons)

mislabeled_EvNum_electrons = list()
mislabeled_EvNum_muons = list()

num_electrons = 13862 # adjust according to the respective file and test sample

for index in mislabeled_index:
    if (index <= num_electrons):
        mislabeled_EvNum_electrons.append(evnum_electrons[index])
    else:
        mislabeled_EvNum_muons.append(evnum_muons[index-num_electrons])

electron_file = open("misclassified_electrons.dat","w")
for evnum in mislabeled_EvNum_electrons:
    electron_file.write("%i \n" % evnum)
electron_file.close()


muon_file = open("misclassified_muons.dat","w")
for evnum in mislabeled_EvNum_muons:
    muon_file.write("%i \n" % evnum)
muon_file.close()
        
     
    


