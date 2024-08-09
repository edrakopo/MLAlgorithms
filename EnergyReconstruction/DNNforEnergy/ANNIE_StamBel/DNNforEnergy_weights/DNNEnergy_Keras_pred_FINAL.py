import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import joblib

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forEnergy.csv"
infile = "shuffled_cut_data.csv"
#infile = "tankPMT_forEnergy_till800MeV.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
df=pd.read_csv(filein)
df = df.iloc[:, 1:]
print(df.head())

Dataset = np.array(df)
lambdas, features, rest, nhits, recoVtxFOM, TrueTrackLengthInMrd, labels = np.split(Dataset,[1100,5500,5505,5506,5507,5508],axis=1)
print("features.shape ", features.shape)
print(features[:2])
print("nhits.shape ", nhits.shape)
print(nhits[:2])
print(TrueTrackLengthInMrd[:2])
print(labels[:2])

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:20000]
train_y = labels[:20000]
test_x = features[20000:]
test_y = labels[20000:]

print(test_x[:5])

#    print("len(train_y): ",len(train_y)," len(test_y): ", len(test_y))
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)
print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
# Load the scaler
scaler = joblib.load('scaler.pkl') 
#train_x = scaler.fit_transform(train_x)

x_transformed = scaler.transform(test_x)
print(x_transformed[:5])

# create model
model = Sequential()
model.add(Dense(20, input_dim=4400, kernel_initializer='normal', activation='leaky_relu'))
model.add(Dense(10, kernel_initializer='normal', activation='leaky_relu'))
model.add(Dense(1, kernel_initializer='normal', activation='leaky_relu'))



# Compile model
model.compile(loss='mean_absolute_error', optimizer='Adamax', metrics=['mean_absolute_error'], weighted_metrics=['mse']) #metrics = ["accuracy"]
print("Created model and loaded weights from file")

# load weights
model.load_weights("model.h5")

## Predict.
print('predicting...')
y_predicted = model.predict(x_transformed)

print("y_pred= ",y_predicted[:5])
print("Edepos= ",test_y[:5])

# Score with sklearn.
score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
print('MSE (sklearn): {0:f}'.format(score_sklearn))

#-----------------------------
print(" saving .csv file with energy variables..")
print("shapes: ", test_y.shape, ", ", y_predicted.shape)

data = np.concatenate((test_y, y_predicted),axis=1)
df=pd.DataFrame(data, columns=['TrueEnergy','DNNEnergy'])

#df.to_csv("Etrue_Ereco.csv", float_format = '%.3f')
df.to_csv("Etrue_Ereco_allenergies_till1000w.csv", float_format = '%.3f')

'''
#---read .csv file containing predict
filein2 = open(str(infile))
df0 = pd.read_csv(filein)[20000:]
#    print(df0.head())
#    df0= pd.read_csv("../LocalFolder/data_forRecoLength_9.csv")
print("df0.shape: ",df0.shape," df.shape: ",df.shape)
print("df0.head(): ",df0.head())
print("df.head(): ", df.head())
#df_final = pd.concat([df0,df], axis=1).drop(['lambda_max.1'], axis=1)
df_final = df0
print("-- Prev: df_final.columns: ",df_final.columns)
df_final.insert(2217, 'TrueTrackLengthInWater', df['TrueTrackLengthInWater'].values, allow_duplicates="True")
df_final.insert(2218, 'DNNRecoLength', df['DNNRecoLength'].values, allow_duplicates="True")
print("df_final.head(): ",df_final.head())
print("--- df_final.shape: ",df_final.shape)
print("-- After: df_final.columns: ",df_final.columns)

#-logical tests:
print("checking..."," df0.shape[0]: ",df0.shape[0]," len(y_predicted): ", len(y_predicted))
assert(df0.shape[0]==len(y_predicted))
print("df_final.shape[0]: ",df_final.shape[0]," df.shape[0]: ",df.shape[0])
assert(df_final.shape[0]==df.shape[0])
'''
#df_final.to_csv("../LocalFolder/vars_Ereco.csv", float_format = '%.3f')
#df_final.to_csv("vars_Ereco_04202019.csv", float_format = '%.3f')
#df_final.to_csv("vars_Ereco_05202019.csv", float_format = '%.3f')
#df_final[:600].to_csv("vars_Ereco_train_05202019.csv", float_format = '%.3f') #to be used for the energy BDT training
#df_final[600:].to_csv("vars_Ereco_pred_05202019.csv", float_format = '%.3f') #to be used for the energy prediction

#df_final.to_csv("vars_Ereco_06082019.csv", float_format = '%.3f')
#df_final[:1000].to_csv("vars_Ereco_train_06082019.csv", float_format = '%.3f')
#df_final[1000:].to_csv("vars_Ereco_pred_06082019.csv", float_format = '%.3f')

#df_final.to_csv("vars_Ereco_06082019CC0pi.csv", float_format = '%.3f')
#df_final[:1000].to_csv("vars_Ereco_train_06082019CC0pi.csv", float_format = '%.3f')
#df_final[1000:].to_csv("vars_Ereco_pred_06082019CC0pi.csv", float_format = '%.3f')

#---if asserts fails check dimensions with these print outs:
#print("df: ",df.head())
#print(df.iloc[:,2200:])
#print(df0.head())
#print(df0.shape)
#print(df0.iloc[:,2200:])
#print(df_final.shape)
#print(df_final.iloc[:,2200:])
