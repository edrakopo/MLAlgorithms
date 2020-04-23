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

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "../data_forRecoLength_04202019.csv"
infile = "../data_forRecoLength_05202019.csv"
#infile = "../LocalFolder/NEWdata_forRecoLength_9_10MRD.csv"
#infile = "../LocalFolder/data_forRecoLength_9.csv"
#--- evts for prediction:
#infile2 = "../data_forRecoLength_04202019.csv"
infile2 = "../data_forRecoLength_05202019.csv"
#infile2 = "../LocalFolder/NEWdata_forRecoLength_0_8MRD.csv"
#infile2 = "../LocalFolder/data_forRecoLength_9.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)
tf.set_random_seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset = np.array(pd.read_csv(filein))
features, lambdamax, labels, rest = np.split(Dataset,[2203,2204,2205],axis=1)

#--- events for predicting
filein2 = open(str(infile2))
print("events for prediction in: ",filein2)
Dataset2 = np.array(pd.read_csv(filein2))
features2, lambdamax2, labels2, rest2 = np.split(Dataset2,[2203,2204,2205],axis=1)
print( "lambdamax2 ", lambdamax2[:2], labels[:2])
print(features2[0])

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:1000]
train_y = labels[:1000]
test_x = features2[1000:]
test_y = labels2[1000:]
lambdamax_test = lambdamax2
#    print("len(train_y): ",len(train_y)," len(test_y): ", len(test_y))
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

#Build 2 layer fully connected DNN with 10, 10 units respectively.
feature_columns = [
    tf.feature_column.numeric_column('x', shape=np.array(train_x).shape[1:])]
regressor = tf.estimator.DNNRegressor(
#    feature_columns=feature_columns, hidden_units=[70, 20])
    feature_columns=feature_columns, hidden_units=[90, 10])

# Train.
print('training....')
batch_size = 2 #1#2
epochs_no= 2000
n_batches = int(np.ceil(num_events / batch_size))
train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_x}, y=train_y, batch_size=batch_size, num_epochs=epochs_no, shuffle=False,num_threads=1)
regressor.train(input_fn=train_input_fn,steps=1000) #1000)

# Predict.
print('predicting...')
x_transformed = scaler.transform(test_x)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_transformed}, y=test_y, shuffle=False)
predictions = regressor.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['predictions'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(test_y).shape)

# Score with sklearn.
score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
print('MSE (sklearn): {0:f}'.format(score_sklearn))

# Score with tensorflow.
scores = regressor.evaluate(input_fn=test_input_fn)
print('MSE (tensorflow): {0:f}'.format(scores['average_loss']))

#-----------------------------
print(" saving .csv file with energy variables..")
print("shapes: ", test_y.shape, ", ", y_predicted.shape)

data = np.concatenate((test_y, y_predicted),axis=1)
df=pd.DataFrame(data, columns=['TrueTrackLengthInWater','DNNRecoLength'])

#---read .csv file containing predict
filein2 = open(str(infile2))
df0 = pd.read_csv(filein2)[1000:]
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

#df_final.to_csv("../LocalFolder/vars_Ereco.csv", float_format = '%.3f')
#df_final.to_csv("vars_Ereco_04202019.csv", float_format = '%.3f')
df_final[:600].to_csv("vars_Ereco_train_05202019.csv", float_format = '%.3f')
df_final[600:].to_csv("vars_Ereco_pred_05202019.csv", float_format = '%.3f')

#---if asserts fails check dimensions with these print outs:
#print("df: ",df.head())
#print(df.iloc[:,2200:])
#print(df0.head())
#print(df0.shape)
#print(df0.iloc[:,2200:])
#print(df_final.shape)
#print(df_final.iloc[:,2200:])

