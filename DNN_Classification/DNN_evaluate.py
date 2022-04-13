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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve


in_dim = 1500

#infileY = "~/git/ANNIETools/ANNIENtupleAnalysis/util/DNN_eval.csv"
#infileD = "~/git/ANNIETools/ANNIENtupleAnalysis/util/DNN_eval_noqt.csv"

infileY = "~/git/ANNIETools/ANNIENtupleAnalysis/util/DNN_mix_grill_all_cb.csv"
infileD = "~/git/ANNIETools/ANNIENtupleAnalysis/util/DNN_mix_grill_all_cb_noqt.csv"

#infileY = "~/git/ANNIETools/ANNIENtupleAnalysis/util/DNN_mix_grill_all_cb_quality_cuts.csv"
#infileD = "~/git/ANNIETools/ANNIENtupleAnalysis/util/DNN_mix_grill_all_cb_noqt_quality_cuts.csv"

#infileD = "~/git/ANNIETools/ANNIENtupleAnalysis/util/run2263_nocuts_noqt.csv"
#infileY = "~/git/ANNIETools/ANNIENtupleAnalysis/util/run2263_nocuts.csv"


dataX = pd.read_csv(infileD, header=None, names=range(in_dim)).fillna(value=0)
print('read X file')
dataY = pd.read_csv(infileY ,header=None, low_memory=False)
print('read Y file')
print('passed step :read csv & cleared ram')


#print(dataX[0])
print(dataX.head())
evalX = dataX.iloc[:, 0:1500]
print('eval shape',evalX.shape)
#print(x_original.shape)
evalY = dataY.iloc[:, 9:10]
#print(x_original.head())
#rest, y_original = np.split(dataY,[9],axis=1) 
evalY = np.array(evalY)
evalY = evalY.reshape(-1)
#print('y_original index is:', len(y_original))
#y_original = np.ravel(dataY['labels'])
print('passed step: split into x/y lists')
#del dataY
#gc.collect()


print('passed step: split into train/test lists')

# normalize the dataset
scaler = StandardScaler()
test_x = scaler.fit_transform(evalX)


# create model
model = Sequential()
model.add(Dense(35, input_dim=in_dim, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
'''

def create_model():
    # create model
    model = Sequential()
    
    model.add(Dense(40, input_dim=in_dim, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
   
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
'''

#keras_model = create_model()

#model.load_weights("Model_cb<0.4_cb>0.4.hdf5")
#model.load_weights("Model_cb<0.4_cb>0.9.hdf5")
#model.load_weights('weights_bets_20_10_ep100.hdf5')
#model.load_weights('weights_bets_20_10.hdf5')
#model.load_weights('weights_bets_20_10_3000.hdf5')
#model.load_weights('weights_bets_60_30_3000.hdf5')
model.load_weights('weights_bets_35_17_1500_125.hdf5')
#model.load_weights('weights_bets_35_17_1500_125_20ep2.hdf5')

## Predict.
print('predicting...')
#estimator = KerasClassifier(build_fn=create_model, epochs=15, batch_size=10, verbose=0)
y_pred = model.predict(test_x).ravel()
print(y_pred)
y_pred = np.vstack(y_pred)
y_pred = y_pred.astype(int)
print(y_pred)
evalY = np.vstack(evalY)
print(evalY)
df1 = pd.DataFrame(data=evalY,columns=['Yvalues'])
df2 = pd.DataFrame(data=y_pred,columns=['Prediction'])
df = pd.concat([df1,df2], axis=1)
print(df.head(5))
df.to_csv("predictionsKeras_evaluate.csv")


'''
Y_probs = model.predict_on_batch(evalX)
#print("testY: ",testY[:10]," and predicts ", Ypred[:10])
#print(type(Xtest)," ",type(testY)," testY.shape ",testY.shape," Ypred.shape ",Ypred.shape)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(evalY, Y_probs[:,0], pos_label=1) #Ypred)
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
plt.savefig("ROC_curve_evaluate.pdf")
plt.show()
'''
print('Final predicted_data_eval.\n.\n.\n.\n.')
#assert(evalX.shape[0]==len(y_pred))
ydata = np.concatenate((evalY,y_pred),axis=1)
df=pd.DataFrame(ydata, columns=['TrueY','Predicted'])
print(df.head())
#df_final = data[3000:]
df_final = dataX
df_final.insert(1500, 'TrueY', df['TrueY'].values, allow_duplicates="True")
df_final.insert(1501, 'Predicted', df['Predicted'].values, allow_duplicates="True")
print(df_final.sample(10))
df_final.to_csv("predicted_data_evaluate.csv", float_format = '%.3f', index=False)


