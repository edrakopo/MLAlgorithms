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

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import gc
#from tensorflow.keras.optimizers import SGD

#------------- events for train, signal & bkgd with CB cuts ------------#

#infileS = "~/git/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_BKGD_CBcuts.csv"
#infileB = "~/git/ANNIETools/ANNIENtupleAnalysis/util/vars_DNN_Signal_BKGD_CBcuts2.csv"

infileS = "~/git/ANNIETools/ANNIENtupleAnalysis/util/small.csv"
infileB = "~/git/ANNIETools/ANNIENtupleAnalysis/util/small2.csv"


# Set TF random seed to improve reproducibility
seed = 125
np.random.seed(seed)

print( "--- opening file with input variables!")

dataX = pd.read_csv(infileB,header=None, names=range(1500)).fillna(value=0)
print('read X file')
dataY = pd.read_csv(infileS,header=None, low_memory=False)
print('read Y file')



#print(dataX[0])
#print(dataX.head(1))
x_original = dataX.iloc[:, 0:1500]
#print(x_original.shape)
y_original = dataY.iloc[:, 9:10]
#print(x_original.head())
#rest, y_original = np.split(dataY,[9],axis=1) 
y_original = np.array(y_original)
y_original = y_original.reshape(-1)
#print('y_original index is:', len(y_original))
#y_original = np.ravel(dataY['labels'])
print('passed step: split into x/y lists')
#del dataY
#gc.collect()

trainX, testX, trainY, testY = train_test_split(x_original, y_original ,test_size=0.3)
print('passed step: split into train/test lists')


# Scale data (training set) to 0 mean and unit standard deviation.
scaler = StandardScaler()
X = scaler.fit_transform(trainX)
print("X.shape: ", X.shape)
print("types: ",type(X), "\n",type(trainY))
Xtest = scaler.fit_transform(testX)
print("testX.shape: ", testX.shape," testY.shape ",testY.shape)



# encode class values as integers ------------------- OLDDDDD -----------

encoder = LabelEncoder()
encoder.fit(trainY)
encoded_Y = encoder.transform(trainY)


def create_model():
    # create model
    model = Sequential()
    #for prompt events
    model.add(Dense(35, input_dim=1500, activation='relu'))
    model.add(Dense(17, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    '''#for delayed events
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

keras_model = create_model()
keras_model.fit(trainX, trainY, epochs=70, batch_size=5, verbose=1)
#estimator = KerasClassifier(build_fn=create_model, epochs=20, batch_size=5, verbose=0)#for prompt events
estimator = KerasClassifier(build_fn=create_model, epochs=15, batch_size=5, verbose=0)
#estimator.fit(X,encoded_Y,verbose=0)



# checkpoint
filepath=f"weights_bets_35_17_1500_{seed}_15ep.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]
# Fit the model
history = estimator.fit(X, encoded_Y, validation_split=0.33, epochs=15, batch_size=5, callbacks=callbacks_list, verbose=0)
#I can add: validation_data=(testX,testy) instead of validation_split

# summarize history for accuracy:
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.set_title('model accuracy')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
ax2.set_ylim(0.5,1.)
ax2.legend(['train', 'test'], loc='upper left')
plt.savefig(f"keras_train_testAcc_{seed}_4.pdf")
plt.show()

# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='test')
ax2.set_title('model loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
ax2.set_ylim(0.,0.5)
ax2.legend(['train', 'test'], loc='upper left')
plt.savefig(f"keras_train_testLoss_{seed}_4.pdf")
plt.show()

#ROC curve:
from sklearn.metrics import roc_curve

Ypred = estimator.predict(Xtest).ravel()
Ypred = np.vstack(Ypred)
testY = np.vstack(testY)
print('testY is: \n',testY)
#store  predictions to csv:
df1 = pd.DataFrame(data=testY,columns=['Yvalues'])
df2 = pd.DataFrame(data=Ypred,columns=['Prediction'])
print("check df head:\n ",df1.head()," \n", df2.head())
df = pd.concat([df1,df2], axis=1)
print(df.head())
df.to_csv("predictionsKeras.csv")

Y_probs = estimator.predict_proba(Xtest)
#print("testY: ",testY[:10]," and predicts ", Ypred[:10])
#print(type(Xtest)," ",type(testY)," testY.shape ",testY.shape," Ypred.shape ",Ypred.shape)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(testY, Y_probs[:,1], pos_label=1) #Ypred)
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
plt.show()

print(metrics.classification_report(testY, Ypred))

#print variables in csv for checks:
assert(testX.shape[0]==len(Ypred))
ydata = np.concatenate((testY,Ypred),axis=1)
df=pd.DataFrame(ydata, columns=['TrueY','Predicted'])
print(df.head())
#df_final = data[1500:]
df_final = dataX[len(trainX):]
df_final.insert(1500, 'TrueY', df['TrueY'].values, allow_duplicates="True")
df_final.insert(1501, 'Predicted', df['Predicted'].values, allow_duplicates="True")
print(df_final.head())
df_final.to_csv("predicted_data.csv", float_format = '%.3f', index=False)

