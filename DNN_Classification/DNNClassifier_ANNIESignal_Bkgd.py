#for code explanation check: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#--------- File with events for reconstruction:
#--- evts for training:
infile = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_Bkgd2.csv"
infile0 = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_Bkgd.csv"
#

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events: variables
#df = pd.read_csv(infile)#,names=range(1000))
#print("All columns are: ", df.columns.values.tolist())
#filein = open(str(infile))
#print("evts for training in: ",filein)
#df=pd.read_csv(infile)
data = pd.read_csv(infile,header=None,names=range(1500)).fillna(0)
print(data.head())
trainX = np.array(data[:300])
#X = np.array(df.drop(['label'], axis=1))
#X, rest = np.split(Dataset,[5],axis=1)
#X = df[:,0:8]#.astype(float)
print("trainX[0]: ",trainX[0])
print(trainX.shape)
data0 = np.array(pd.read_csv(infile0)[:300])
rest, Y = np.split(data0,[6],axis=1)
print("before Y.shape ",Y.shape)
Y=Y.reshape(-1)
print("after: ",Y.shape)
print("Y: ",Y[0])
'''
#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
train_x = features[:1000]
train_y = labels[:1000]
print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)
'''
# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(trainX)
print("X.shape: ", X.shape)
print("types: ",type(X), " ",type(Y))

'''
# load dataset
dataframe = pd.read_csv("sonar.csv", header=None)
dataset = dataframe.values
print(dataframe.head())
print(dataframe.tail())
# split into input (X) and output (Y) variables
X1 = dataset[:,0:60].astype(float)
Y1 = dataset[:,60]
print("types1: ",type(X1), " ",type(Y1))
print("X1.shape ",X1.shape)
print("Y1.shape ",Y1.shape)
'''
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(800, input_dim=1500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
'''
#evaluate this model using stratified cross validation in the scikit-learn
estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

# evaluate baseline model using a pipeline with: i)stratified cross validation and ii)a standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


