import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import pyplot
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#set random number to improve reproducibility
seed = 170
np.random.seed(seed)

Signal_file = "anal_scripts_ROOT_CSV_py/vars_Signal_eval.csv"
Bkgd_file = "anal_scripts_ROOT_CSV_py/vars_Bkgd_eval.csv"

columns = ["TrLengthIT_2","zenith","ratio_doms","ratio_casc_doms","jlik","logbeta0","reco_cheredoms","ratio_pmts","logEreco","jz_f","D","max_zen_sol","ratio330","ratio6","ratio4","prefitN_1degree","prefitN_up","max_diff_sollik","redToT_cascmuon","diff_dist_mu","ratio310","ratiol", "label"]

#get signal data
sig = pd.read_csv(Signal_file, usecols=columns)
#print(sig.head())
print(sig.shape)
#get Bkgd data
bkg = pd.read_csv(Bkgd_file, usecols=columns)
#print(bkg.head())
print(bkg.shape)

#Merge signal and Bkgd
data0 = pd.concat([sig,bkg],axis=0)
print(data0.head())

#randomly shuffle the data
data = shuffle(data0, random_state=0) 
print("after shuffling: ", data.head())
print("data.shape: ", data.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_eval = pd.DataFrame(scaler.fit_transform(data.iloc[:,:22]))

Y_eval = data.iloc[:,22]
print("type(X_eval) ",type(X_eval)," type(Y_eval): ",type(Y_eval))
print("X_eval.shape: ",X_eval.shape," Y_eval.shape: ",Y_eval.shape)

# load the model from disk
filename = 'xgboost_Classify.sav' 
loaded_model = pickle.load(open(filename, 'rb'))

#predicting...
y_pred = loaded_model.predict(X_eval)
print(y_pred[:10])
predictions = [round(value) for value in y_pred]
print(predictions[:10])

print("Y_eval: ",Y_eval[:10]," y_pred: ",y_pred[:10])

accuracy0 = accuracy_score(Y_eval, predictions)
print("accuracy0: ",accuracy0)

# calculate the accuracy score
accuracy =  accuracy_score(Y_eval, y_pred) * 100 #returns the fraction of correctly classified samples
print('Accuracy: %.3f' % accuracy)
print("Y_eval: ",Y_eval.head()," y_pred: ",pd.DataFrame(y_pred).head())
print("Y_eval: ",Y_eval[:20]," y_pred: ",y_pred[:20])
print("-------------------------------------")
predictions0 = pd.concat([Y_eval.reset_index(drop=True) ,pd.DataFrame(y_pred,columns=['Prediction'])], axis=1)
print(predictions0.head())
predictions0.to_csv( 'predictions_XGBoost_3percent.csv')

report = classification_report(Y_eval, y_pred)
print(report)

#Calculate and Plot ROC curve
from sklearn.metrics import roc_curve, auc

loaded_model.probability = True
probas = loaded_model.predict_proba(X_eval)
fpr, tpr, thresholds = roc_curve(Y_eval, probas[:, 1], pos_label =1)# 1 is signal
roc_auc  = auc(fpr, tpr)

fig2 = plt.figure(1,figsize=(15,20)) 
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ("XGBoost", roc_auc))
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc=0, fontsize='small')
 
fig2 = plt.figure(1,figsize=(15,20))

plt.show()
