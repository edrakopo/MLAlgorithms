import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

### read in the accuracy scores

labels = ["all parameters","no distHor","no distVert","no dists","no Var","no Skew","no Kurt","no VarSkewKurt","no distsVarSkewKurt"]
labels_classifier = ["Decision Tree","Random Forest","XGBoost","SVM","NN","Gradient Boosting","MLP"]
filenames = ["accuracy/Accuracy_AllParameters.dat","accuracy/Accuracy_NoDistHor.dat","accuracy/Accuracy_NoDistVert.dat","accuracy/Accuracy_NoDists.dat","accuracy/Accuracy_NoVar.dat","accuracy/Accuracy_NoSkew.dat","accuracy/Accuracy_NoKurt.dat","accuracy/Accuracy_NoVarSkewKurt.dat","accuracy/Accuracy_NoDistsNoSkew.dat"]

accuracyDecisionTree = [0]*9
accuracyRandomForest = [0]*9
accuracyXGBoost = [0]*9
accuracySVM = [0]*9
accuracyNearestN = [0]*9
accuracySGD = [0]*9
accuracyGNB = [0]*9
accuracyMLP = [0]*9
accuracyLogistic = [0]*9
accuracyGradientBoosting = [0]*9

for i in range(9):
    file = open(filenames[i],'r')
    f1 = file.readlines()
    j=0
    for x in f1:
        if (j==0):
            accuracyDecisionTree[i] = float(x)
        elif (j==1):
            accuracyRandomForest[i] = float(x)
        elif (j==2):
            accuracyXGBoost[i] = float(x)
        elif (j==3):
            accuracySVM[i] = float(x)
        elif (j==4):
            accuracyNearestN[i] = float(x)
        elif (j==5):
            accuracySGD[i] = float(x)
        elif (j==6):
            accuracyGNB[i] = float(x)
        elif (j==7):
            accuracyMLP[i] = float(x)
        elif (j==8):
            accuracyLogistic[i] = float(x)
        elif (j==9):
            accuracyGradientBoosting[i] = float(x)
        j=j+1
    file.close()

print("Accuracies read from file(s):")
print(accuracyDecisionTree)         #exemplary printout for Decision Tree classifier

x_values = np.arange(0,9,1.0)

plt.plot(x_values,accuracyDecisionTree,'kH',x_values,accuracyRandomForest,'bs',x_values,accuracyXGBoost,'g^',x_values,accuracySVM,'mo',x_values,accuracyNearestN,'r+',x_values,accuracyGradientBoosting,'kv',x_values,accuracyMLP,'cx')
plt.ylabel('accuracy (%)')
plt.xlabel('variable selection')
plt.xticks(range(9),labels,rotation=45,fontsize=7)
plt.legend(labels_classifier)
plt.show()

