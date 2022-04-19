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
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')
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

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import gc
#from tensorflow.keras.optimizers import SGD

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_Bkgd2.csv"
#infile0 = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_Bkgd.csv"
#infile = "/Users/edrakopo/work/ANNIETools_ntuples/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_BkgdNEW2.csv"
#infile0 = "/Users/edrakopo/work/ANNIETools_ntuples/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_BkgdNEW.csv"
 
#--- prompt evts for signal:
#infile = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_Bkgd_prompt2.csv"
#infile0 = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_Bkgd_prompt.csv"

#--- delayed evts for signal+Bkgd:
#infile = "~/git/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_Bkgd_delNEW2.csv"
#infile0 = "~/git/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_Bkgd_delNEW.csv"

#------------- events for train, signal & bkgd with CB cuts ------------#

#infileS = "~/git/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_BKGD_CBcuts.csv"
#infileB = "~/git/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_BKGD_CBcuts2.csv"

infileS = "~/git/ANNIETools/ANNIENtupleAnalysis/util/small.csv"
infileB = "~/git/ANNIETools/ANNIENtupleAnalysis/util/small2.csv"




# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events: variables
#print("All columns are: ", df.columns.values.tolist())
#filein = open(str(infile))
#print("evts for training in: ",filein)
#dataX = pd.read_csv(infile,header=None,names=range(3000)).fillna(0)
#print(data.head())
#trainX = np.array(data[:3000]) #variables to train
#testX = np.array(data[3000:]) #variables to test
#trainX = np.array(data[:30000]) #variables to train
#testX = np.array(data[30000:]) #variables to test
#print("trainX[0]: ",trainX[0])
#print(trainX.shape)
#print("testX[0]: ",testX[0])
#print("All columns are: ", data.columns.values.tolist())

'''
dataY00 = pd.read_csv(infile0)
#dataY0 = np.array(dataY00[:3000])
dataY0 = np.array(dataY00[:30000])
#rest, Y = np.split(dataY0,[6],axis=1)
rest, Y = np.split(dataY0,[7],axis=1)
print("before Y.shape ",Y.shape)
Y=Y.reshape(-1)
print("after: ",Y.shape)
print("Y: ",Y[0:10])
'''

dataX = pd.read_csv(infileB,header=None, names=range(1500)).fillna(value=0)
print('read X file')
dataY = pd.read_csv(infileS,header=None, low_memory=False)
print('read Y file')
gc.collect()
print('passed step :read csv & cleared ram')


#print(dataX[0])
print(dataX.head(1))
x_original = dataX.iloc[:50000, 0:1500]
#print(x_original.sample(10))
y_original = dataY.iloc[:50000, 9:10]
#print(x_original.head())
y_original = np.array(y_original)
y_original = y_original.reshape(-1)
Y= y_original
#print('y_original index is:', len(y_original))
#y_original = np.ravel(dataY['labels'])
print('passed step: split into x/y lists')


#trainX, testX, trainY, testY = train_test_split(x_original, y_original ,test_size=0.3)
print('passed step: split into train/test lists')
'''
#datatestY = np.array(dataY00[3000:])
datatestY = np.array(dataY00[30000:])
#rest1, testY = np.split(datatestY,[6],axis=1)
rest1, testY = np.split(datatestY,[7],axis=1)
testY=testY.reshape(-1)
'''

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = StandardScaler()
#X = scaler.fit_transform(trainX)
X = scaler.fit_transform(x_original)
print("X.shape: ", X.shape)
#print("types: ",type(X), "\n",type(trainY))
#Xtest = scaler.fit_transform(testX)
print("testX.shape: ", X.shape," testY.shape ",Y.shape)

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

'''
# encode class values as integers ------------------- OLDDDDD -----------
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
'''
#encoder = LabelEncoder()
#encoder.fit(trainY)
#encoded_Y = encoder.transform(trainY)


def create_model(neurons1=30, neurons2=15):
    # create model
    model = Sequential()
    #for prompt events
    model.add(Dense(neurons1, input_dim=1500, activation='relu'))
    model.add(Dense(neurons2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    '''#for delayed events
    model.add(Dense(40, input_dim=3000, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    '''
    # Compile model
    #opt=SGD(lr=0.01, momentum=0.9)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0, batch_size=10, epochs=20)
epochs=[8,12,15,20]
batch_size=[5,10,15]
learning_rate = [0.001, 0.01]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
neurons1 = [20,30, 40, 50]
neurons2 = [10,15, 20, 25]
#param_grid = dict(epochs=epochs, batch_size=batch_size, learning_rate = learning_rate)#, momentum = momentum)
param_grid = dict(neurons1=neurons1, neurons2=neurons2)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=6, cv=3)
grid_result = grid.fit(X,Y)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''
# summarize history for accuracy:
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['accuracy'])
#ax2.plot(history.history['val_accuracy'])
ax2.set_title('model accuracy')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
#ax2.set_xlim(0.,10.)
#ax2.legend(['train', 'test'], loc='best')
#plt.savefig("keras_train_testAcc.pdf")
#plt.show()

# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'])#, label='train')
#ax2.plot(history.history['val_loss'], label='test')
ax2.set_title('model loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
#ax2.set_xlim(0.,10.)
#ax2.legend(['train', 'test'], loc='best')
#plt.savefig("keras_train_testLoss.pdf")
plt.show()
'''
