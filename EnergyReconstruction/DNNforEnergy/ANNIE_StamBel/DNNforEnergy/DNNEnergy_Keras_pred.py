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
import os
from sklearn.utils import shuffle


# Set the directory containing the CSV files
directory = "/home/tinabel/Desktop/ToolAnalysisLink/DNNforEnergy/EnergyReconstruction/DNNforEnergy/tankPMT_forEnergy" #alex files. Use a few for train and the rest for pred.
infile = "/home/tinabel/Desktop/ToolAnalysisLink/DNNforEnergy/EnergyReconstruction/DNNforEnergy/tankPMT_forEnergy/bigfile/tankPMT_forEnergy_bigfile.csv" #bigfile. Use all for train.

#Dataframe for alex files.Choose the first 4000 events.
filein = open(str(infile))
df1=pd.read_csv(filein)
df2=df1[:5000]
df3=df1[5000:] #The ones we keep for testing.
print(df2.shape)

# Initialize an empty list to store DataFrames
dataframes = []

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        print(f"Reading file: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Shape of DataFrame read from {filename}: {df.shape}")
        dataframes.append(df)

dataframes.append(df2)

# Concatenate all DataFrames into a single DataFrame
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Print the shape of the concatenated DataFrame
print(f"Shape of concatenated DataFrame: {concatenated_df.shape}")

# Convert the concatenated DataFrame to a NumPy array --TRAINING
data = shuffle(concatenated_df, random_state=0)
data=data[data["trueKE"]<1000]
Dataset = data.to_numpy()

# Convert the concatenated DataFrame to a NumPy array --TESTING
data3 = shuffle(df3, random_state=0)
data3=data3[data3["trueKE"]<1000]
Dataset3 = data3.to_numpy()


print( "--- opening file with input variables! --TESTING")
#--- events for training - MC events
#filein = open(str(infile))
#print("evts for training in: ",filein)
#Dataset = np.array(pd.read_csv(filein))
lambdas3, features3, rest3, nhits3, TrueTrackLengthInMrd3, labels3 = np.split(Dataset3,[1100,5500,5505,5506,5507],axis=1)
print("features.shape ", features3.shape)
print(features3[:2])
print("nhits.shape ", nhits3.shape)
print(nhits3[:2])
print(TrueTrackLengthInMrd3[:2])
print(labels3[:2])

#Test samples
num_events, num_pixels = features3.shape
print(num_events, num_pixels)
np.random.seed(0)
test_x = features3
test_y = labels3



print( "--- opening file with input variables! --TRAINING")
#--- events for training - MC events
#filein = open(str(infile))
#print("evts for training in: ",filein)
#Dataset = np.array(pd.read_csv(filein))
lambdas, features, rest, nhits, TrueTrackLengthInMrd, labels = np.split(Dataset,[1100,5500,5505,5506,5507],axis=1)
print("features.shape ", features.shape)
print(features[:2])
print("nhits.shape ", nhits.shape)
print(nhits[:2])
print(TrueTrackLengthInMrd[:2])
print(labels[:2])

#Train Samples
train_x = features
train_y = labels
#--- events for predicting
#filein2 = open(str(infile))
#print("events for prediction in: ",filein2)
#Dataset2 = np.array(pd.read_csv(filein2))
#features2, lambdamax2, labels2, rest2 = np.split(Dataset2,[2203,2204,2205],axis=1)
#print( "lambdamax2 ", lambdamax2[:2], labels[:2])
#print(features2[0])

#    print("len(train_y): ",len(train_y)," len(test_y): ", len(test_y))
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)
print("test sample features shape: ", test_x.shape," test sample label shape: ", test_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

# create model
model = Sequential()
model.add(Dense(18, input_dim=4400, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='relu'))

# load weights
model.load_weights("weights_bets_after_phase_till1000.hdf5")

# Compile model
model.compile(loss='mean_absolute_error', optimizer='Adamax', metrics=['accuracy'])
print("Created model and loaded weights from file")

## Predict.
print('predicting...')
x_transformed = scaler.transform(test_x)
y_predicted = model.predict(x_transformed)

# Score with sklearn.
score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
print('MSE (sklearn): {0:f}'.format(score_sklearn))

#-----------------------------
print(" saving .csv file with energy variables..")
print("shapes: ", test_y.shape, ", ", y_predicted.shape)

data = np.concatenate((test_y, y_predicted),axis=1)
df=pd.DataFrame(data, columns=['TrueEnergy','DNNEnergy'])

#df.to_csv("Etrue_Ereco.csv", float_format = '%.3f')
df.to_csv("Etrue_Ereco_after_phase_till1000.csv", float_format = '%.3f')

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

