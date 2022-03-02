import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import pyplot
import pickle

Signal_file = "anal_scripts_ROOT_CSV_py/vars_Signal.csv"
Bkgd_file = "anal_scripts_ROOT_CSV_py/vars_Bkgd.csv"

columns = ["TrLengthIT_2","zenith","jlik","logbeta0","ratio4","reco_cheredoms","D","diff_dist_mu"]

#get signal data
sig = pd.read_csv(Signal_file, usecols=columns)
print(sig.head())
print(sig.shape)
#get Bkgd data
bkg = pd.read_csv(Bkgd_file, usecols=columns)
print(bkg.head())
print(bkg.shape)

Signal_file1 = "anal_scripts_ROOT_CSV_py/vars_Signal_eval.csv"
Bkgd_file1 = "anal_scripts_ROOT_CSV_py/vars_Bkgd_eval.csv"

#get signal data
sig1 = pd.read_csv(Signal_file1, usecols=columns)
print(sig1.shape)
#get Bkgd data
bkg1 = pd.read_csv(Bkgd_file1, usecols=columns)
print(bkg1.shape)

#Merge signal and Bkgd
dflist = [sig,sig1]
data0 = pd.concat(dflist,axis=0)
print(data0.head())
print("Full data sample",data0.shape)

dfBKglist = [bkg,bkg1]
data0_Bkg = pd.concat(dfBKglist,axis=0)
print("Full data sample",data0_Bkg.shape)

#randomly shuffle the data
data = shuffle(data0, random_state=0) 
print("after shuffling: ", data.head())
print("FULL signal data.shape: ", data.shape)

dataB = shuffle(data0_Bkg, random_state=0) 
print("after shuffling: ", dataB.head())
print("FULL signal dataB.shape: ", dataB.shape)

from sklearn.model_selection import train_test_split
X_train, X_eval = train_test_split(data.iloc[:,:8], test_size = 0.25, random_state = 100)
X_train.to_csv('Trainsample.csv')
X_eval.to_csv('Evalsample.csv')

X_trainB, X_evalB = train_test_split(dataB.iloc[:,:8], test_size = 0.25, random_state = 100)
X_trainB.to_csv('TrainsampleB.csv')
X_evalB.to_csv('EvalsampleB.csv')


