##### Script Track Length Reconstruction in the water tank 
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
#from sklearn import model_selection
from sklearn import preprocessing

#import ROOT
#ROOT.gROOT.SetBatch(True)
#from ROOT import TFile, TNtuple
#from root_numpy import root2array, tree2array, fill_hist

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "data/NEWdata_forRecoLength_9_10MRD.csv"
#infile = "../LocalFolder/data_forRecoLength_9.csv"
#--- evts for prediction:
#infile2 = "data/NEWdata_forRecoLength_0_8MRD.csv" 
#infile2 = "../LocalFolder/data_forRecoLength_9.csv"
#

#Select verbosity! 
verbose=0

#--- Read data from RingEvents.csv
infile = "RingEvents.csv"
filein = open(str(infile))
print("data in: ",filein)
df0=pd.read_csv(filein, names=["Detector type","Radius in m","Angle in rad","Height in m","x in m","y in m","z in m","Ring number"])
print(df0.head())
#print(df0[210:220])
print("index vals: ",df0.index.values," Columns names: ",df0.columns.values)

#--- group per event: 
index_evt = df0.index[df0['Detector type'] == 'Event number'].tolist()
if verbose==1:
   print(len(index_evt), " index_evt: ",index_evt)
for i in range(len(index_evt)):
    if verbose==1:
       print("event:", i ," index:",index_evt[i])
    if index_evt[i]==index_evt[-1]:
       dfs = df0[index_evt[i]:]
    else:
       dfs = df0[index_evt[i]:index_evt[i+1]]   
    #print("dfs: ",dfs)
    #--- drop these lines:
    skip_lines = dfs.index[dfs['Detector type'] == 'Event number'].tolist() + dfs.index[dfs['Detector type'] == 'Detector type'].tolist()
    #print("skip lines: ",dfs.index[dfs['Detector type'] == 'Event number'].tolist())
    df_evt = dfs.drop(skip_lines)
    #print("df_evt: ",df_evt)
    
    #drop some columns from the data sample
    #X  = df_evt.drop(["Detector type", "Ring number", "Radius in m","Angle in rad","Height in m"], axis=1).values
    #X0  = df_evt.drop(["Detector type", "Ring number", "Radius in m","Angle in rad","Height in m"], axis=1)
    X0  = df_evt.drop(["Detector type", "Ring number"], axis=1)
    X0['Radius in m'] = X0['Radius in m'].astype('float64')
    X0['Angle in rad'] = X0['Angle in rad'].astype('float64')
    X0['Height in m'] = X0['Height in m'].astype('float64')
    X0['x in m'] = X0['x in m'].astype('float64')
    X0['y in m'] = X0['y in m'].astype('float64')
    X0['z in m'] = X0['z in m'].astype('float64')
    X = X0.values
    #print("Columns names @ X: ",X0.columns.values)
    #if i==1:
    #   print(type(X[0])," ",X)
    if verbose==1: 
       print("X.shape: ",X.shape," type(X)",type(X)," len(X) ", len(X))
    #store true labels
    labels_true = df_evt["Ring number"].values
    nrings = max(labels_true)
    if verbose==1: 
       print("labels_true.shape: ",labels_true.shape," type(labels_true): ",type(labels_true))
       print("______________ New Event:", i ," index: ",index_evt[i]," True Number of Rings: ", nrings,"______________")
    Y = labels_true.astype('int')

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)
print("len(X): ",len(X))

#---- random split of events ----
rnd_indices = np.random.rand(len(X)) < 0.50
#--- select events for training/test:
train_x = X[rnd_indices]
train_y = nrings[rnd_indices]
#--- select events for prediction: 
test_x = X[~rnd_indices]
test_y = nrings[~rnd_indices]

num_events, num_pixels = train_x.shape

#split events in train/test samples: 
#np.random.seed(0)
#train_x = features
#train_y = labels
#test_x = features2
#test_y = labels2
#    print("len(train_y): ",len(train_y)," len(test_y): ", len(test_y))
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)
 
#Build 2 layer fully connected DNN with 10, 10 units respectively.
feature_columns = [
   tf.feature_column.numeric_column('x', shape=np.array(train_x).shape[1:])]
regressor = tf.estimator.DNNRegressor(
   feature_columns=feature_columns, hidden_units=[70, 20])

# Train.
print('training....')
batch_size = 1#2
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

fig, ax = plt.subplots()
ax.scatter(test_y,y_predicted)
ax.plot([test_y.min(),test_y.max()],[test_y.min(),test_y.max()],'k--',lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
#plt.show()
plt.savefig("test_recolength.png")



#data = np.concatenate((test_y, y_predicted),axis=1)
#df=pd.DataFrame(data, columns=['TrueTrackLengthInWater','DNNRecoLength'])

##---read .csv file containing predict
#filein2 = open(str(infile2))
#df0 = pd.read_csv(filein2)
##    print(df0.head())
##    df0= pd.read_csv("../LocalFolder/data_forRecoLength_9.csv")
#df_final = pd.concat([df0,df], axis=1).drop(['lambda_max.1'], axis=1)

#-logical tests:
#print("checking..."," df0.shape[0]: ",df0.shape[0]," len(y_predicted): ", len(y_predicted))
#assert(df0.shape[0]==len(y_predicted))
#assert(df_final.shape[0]==df.shape[0])

#df_final.to_csv("data/vars_Ereco.csv", float_format = '%.3f')



