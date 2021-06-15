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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler

#--------- File with events for reconstruction:
#--- evts for prediction:
infile = "seeds_fixedpos_pred.csv"
#infile = "seeds_randompos_large.csv"

# Set random seeds to improve reproducibility
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)

### Loop the data lines and fillin with nan if column is missing
with open(str(infile)) as temp_f:
    # get No of columns in each line
    col_count = [ len(l.split(",")) for l in temp_f.readlines() ]
### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
column_names = [i for i in range(0, max(col_count))]

### Read csv
dataIN = pd.read_csv(filein, header=None, delimiter=",", names=column_names).fillna(0)
print(dataIN.head())
Dataset = np.array(dataIN)
#nhits, labels, features0 = np.split(Dataset,[1,4],axis=1)
#features = np.concatenate((nhits, features0),axis=1)
labels, features = np.split(Dataset,[3],axis=1)
print(features.shape)
in_dim = features.shape[1]
out_dim = labels.shape[1]

#split events in train/test samples:
num_events, num_pixels = features.shape
print(num_events, num_pixels)
np.random.seed(0)
test_x0 = features[0:]
test_y = labels[0:]
#test_x0 = features[25000:]
#test_y = labels[25000:]
print("test_y: ", test_y[:4])

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
test_x = scaler.fit_transform(test_x0)

# create model
model = Sequential()
model.add(Dense(100, input_dim=in_dim, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(out_dim, kernel_initializer='normal', activation='relu'))

# load weights
model.load_weights("weights_bets.hdf5")
#model.load_weights("weights_bets_fixedpos.hdf5")

## Predict.
print('predicting...')
y_predicted = model.predict(test_x)

# Score with sklearn.
score_sklearn = metrics.mean_squared_error(y_predicted, test_y)
print('MSE (sklearn): {0:f}'.format(score_sklearn))

#-----------------------------
print(" saving .csv file with output variables..")
print("shapes: ", test_y.shape, ", ", y_predicted.shape)

data = np.concatenate((test_y, y_predicted),axis=1)
df=pd.DataFrame(data, columns=['mc_x','mc_y','mc_z','reco_x','reco_y','reco_z'])
print(df.head())

df['dR']=np.sqrt( ((df['mc_x']-df['reco_x'])*(df['mc_x']-df['reco_x'])) +  ((df['mc_y']-df['reco_y'])*(df['mc_y']-df['reco_y'])) + ((df['mc_z']-df['reco_z'])*(df['mc_z']-df['reco_z'])))
print(df.head())

nbins=np.arange(0.,100.,0.1)
fig2,ax2=plt.subplots(ncols=1, sharey=False)#, figsize=(8, 6))
f0=ax2.hist(df['dR'], nbins, histtype='step', fill=False, color='blue',alpha=0.75)#, log=True)
#f1=ax2.hist(dataprev, nbins, histtype='step', fill=False, color='red',alpha=0.75)
#ax.set_xlim(0.,200.)
ax2.set_xlabel('$\Delta R$ [cm]')
#ax2.legend(('NEW','Previous'))
#ax2.xaxis.set_ticks(np.arange(0., 425., 25))
ax2.tick_params(axis='x', which='minor', bottom=False)
title = "mean = %.2f, std = %.2f " % (df['dR'].mean(), df['dR'].std())
plt.title(title)
plt.show()
fig2.savefig('dR.png')
plt.close(fig2)

