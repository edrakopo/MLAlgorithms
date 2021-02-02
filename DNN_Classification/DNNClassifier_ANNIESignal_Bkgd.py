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
matplotlib.use('TkAgg')
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
#from tensorflow.keras.optimizers import SGD

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_Bkgd2.csv"
#infile0 = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_Bkgd.csv"
infile = "/Users/edrakopo/work/ANNIETools_ntuples/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_BkgdNEW2.csv"
infile0 = "/Users/edrakopo/work/ANNIETools_ntuples/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_BkgdNEW.csv"
 
#--- prompt evts for signal:
#infile = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_Bkgd_prompt2.csv"
#infile0 = "/Users/edrakopo/work/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_Bkgd_prompt.csv"
#--- delayed evts for signal+Bkgd:
infile = "/Users/edrakopo/work/ANNIETools_ntuples/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_Bkgd_delNEW2.csv"
infile0 = "/Users/edrakopo/work/ANNIETools_ntuples/ANNIETools/ANNIENtupleAnalysis/util/labels_DNN_Signal_Bkgd_delNEW.csv"

# Set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events: variables
#print("All columns are: ", df.columns.values.tolist())
#filein = open(str(infile))
#print("evts for training in: ",filein)
data = pd.read_csv(infile,header=None,names=range(1500)).fillna(0)
print(data.head())
#trainX = np.array(data[:3000]) #variables to train
#testX = np.array(data[3000:]) #variables to test
trainX = np.array(data[:10000]) #variables to train
testX = np.array(data[10000:]) #variables to test
print("trainX[0]: ",trainX[0])
print(trainX.shape)
print("testX[0]: ",testX[0])
#print("All columns are: ", data.columns.values.tolist())

dataY00 = pd.read_csv(infile0)
#dataY0 = np.array(dataY00[:3000])
dataY0 = np.array(dataY00[:10000])
#rest, Y = np.split(dataY0,[6],axis=1)
rest, Y = np.split(dataY0,[7],axis=1)
print("before Y.shape ",Y.shape)
Y=Y.reshape(-1)
print("after: ",Y.shape)
print("Y: ",Y[0:10])

#dataYtest = np.array(dataY00[3000:])
dataYtest = np.array(dataY00[10000:])
#rest1, Ytest = np.split(dataYtest,[6],axis=1)
rest1, Ytest = np.split(dataYtest,[7],axis=1)
Ytest=Ytest.reshape(-1)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(trainX)
print("X.shape: ", X.shape)
print("types: ",type(X), " ",type(Y))
Xtest = scaler.fit_transform(testX)
print("testX.shape: ", testX.shape," Ytest.shape ",Ytest.shape)

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
    #for prompt events
    model.add(Dense(50, input_dim=1500, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    '''#fordelayed events
    model.add(Dense(40, input_dim=1500, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    '''
    # Compile model
    #opt=SGD(lr=0.01, momentum=0.9)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#keras_model = create_model()
#keras_model.fit(trainX, Y, epochs=100, batch_size=10, verbose=1)
#estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size=5, verbose=0)#for prompt events
estimator = KerasClassifier(build_fn=create_model, epochs=15, batch_size=10, verbose=0)
#estimator.fit(X,encoded_Y,verbose=0)
'''
#evaluate this model using stratified cross validation in the scikit-learn
#estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#pred_results = cross_val_score(estimator, Xtest, Ytest, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (pred_results.mean()*100, pred_results.std()*100))
'''
'''
# evaluate baseline model using a pipeline with: i)stratified cross validation and ii)a standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''
# checkpoint
filepath="weights_bets.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]
# Fit the model
history = estimator.fit(X, encoded_Y, validation_split=0.33, epochs=30, batch_size=5, callbacks=callbacks_list, verbose=0)
#I can add: validation_data=(testX,testy) instead of validation_split

# summarize history for accuracy:
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.set_title('model accuracy')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
#ax2.set_xlim(0.,10.)
ax2.legend(['train', 'test'], loc='upper left')
plt.savefig("keras_train_testAcc.pdf")

# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='test')
ax2.set_title('model loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
#ax2.set_xlim(0.,10.)
#ax2.legend(['train', 'test'], loc='upper left')
plt.savefig("keras_train_testLoss.pdf")

#ROC curve:
from sklearn.metrics import roc_curve

Ypred = estimator.predict(Xtest).ravel()
Ypred =np.vstack(Ypred)
Ytest=np.vstack(Ytest)
#store predictions to csv:
df1 = pd.DataFrame(data=Ytest,columns=['Yvalues'])
df2 = pd.DataFrame(data=Ypred,columns=['Prediction'])
print("check df head: ",df1.head()," ", df2.head())
df = pd.concat([df1,df2], axis=1)
print(df.head())
df.to_csv("predictionsKeras.csv")

Y_probs = estimator.predict_proba(Xtest)
#print("Ytest: ",Ytest[:10]," and predicts ", Ypred[:10])
#print(type(Xtest)," ",type(Ytest)," Ytest.shape ",Ytest.shape," Ypred.shape ",Ypred.shape)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Ytest, Y_probs[:, 1], pos_label =1) #Ypred)
#AUC:
from sklearn.metrics import auc

auc_keras = auc(fpr_keras, tpr_keras)

f1, ax1 = plt.subplots(1,1)
ax1.plot([0, 1], [0, 1], 'k--')
ax1.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
ax1.set_xlabel('False positive rate')
ax1.set_ylabel('True positive rate')
ax1.set_title('ROC curve')
ax1.legend(loc='best')
plt.savefig("ROC_curve.pdf")
#plt.show()

print(metrics.classification_report(Ytest, Ypred))

#print variables in csv for checks:
assert(testX.shape[0]==len(Ypred))
ydata = np.concatenate((Ytest,Ypred),axis=1)
df=pd.DataFrame(ydata, columns=['TrueY','Predicted'])
print(df.head())
#df_final = data[3000:]
df_final = data[10000:]
df_final.insert(1500, 'TrueY', df['TrueY'].values, allow_duplicates="True")
df_final.insert(1501, 'Predicted', df['Predicted'].values, allow_duplicates="True")
print(df_final.head())
df_final.to_csv("predicted_data.csv", float_format = '%.3f', index=False)

