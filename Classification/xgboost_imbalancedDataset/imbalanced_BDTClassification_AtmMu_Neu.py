import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import pyplot
import pickle

#Signal_file = "anal_scripts_ROOT_CSV_py/vars_Signal.csv"
#Bkgd_file = "anal_scripts_ROOT_CSV_py/vars_Bkgd.csv"
Signal_file = "Trainsample.csv"
Bkgd_file = "TrainsampleB.csv"

#columns = ["TrLengthIT_2","zenith","ratio_doms","ratio_casc_doms","jlik","logbeta0","reco_cheredoms","ratio_pmts","logEreco","jz_f","D","max_zen_sol","ratio330","ratio6","ratio4","prefitN_1degree","prefitN_up","max_diff_sollik","redToT_cascmuon","diff_dist_mu","ratio310","ratiol", "label"]
columns = ["TrLengthIT_2","zenith","jlik","logbeta0","ratio4","reco_cheredoms","D","diff_dist_mu"]

#get signal data
sig = pd.read_csv(Signal_file, usecols=columns)
sig['newlabel'] = 0
print(sig.head())
print(sig.shape)
#get Bkgd data
bkg = pd.read_csv(Bkgd_file, usecols=columns)
bkg['newlabel'] = 1
print(bkg.head())
print(bkg.shape)

#Merge signal and Bkgd
data0 = pd.concat([sig,bkg],axis=0)
print(data0.head())
data0["TrLengthIT_2"]=data0["TrLengthIT_2"]/1000.
data0["zenith"]=data0["zenith"]/180.
data0["jlik"]=data0["jlik"]/1000.
data0["logbeta0"]=data0["logbeta0"]/4.
data0["ratio4"]=data0["ratio4"]/8.
data0["reco_cheredoms"]=data0["reco_cheredoms"]/50.
data0["D"]=data0["D"]/600.
data0["diff_dist_mu"]=data0["diff_dist_mu"]/10000.
print("checking vars: ", data0.head())

#randomly shuffle the data
data = shuffle(data0, random_state=0) 
print("after shuffling: ", data.head())
print("data.shape: ", data.shape)
print("describe df: ", data.describe())

i_eval = pd.DataFrame(data.iloc[:,:8])
# build train and test dataset
from sklearn.model_selection import train_test_split
#X_train0, X_test0, y_train, y_test = train_test_split(data.iloc[:,:22], data.iloc[:,23], test_size = 0.3, random_state = 100)
X_train0, X_test0, y_train, y_test = train_test_split(data.iloc[:,:8], data.iloc[:,8], test_size = 0.3, random_state = 100)
print(X_train0.head(), " y: ", y_train.head())
print(X_test0.head(), " ytest: ", y_test.head())

# Scale data (training set) to 0 mean and unit standard deviation.
#scaler = preprocessing.StandardScaler()
#print("X_train0 before scaling: ",X_train0.head())
#X_train = pd.DataFrame(scaler.fit_transform(X_train0))
#X_test = pd.DataFrame(scaler.transform(X_test0))
#print("X_train after scaling: ",X_train.head())

X_train = pd.DataFrame(X_train0)
X_test = pd.DataFrame(X_test0)

print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))
print("X_train.shape: ",X_train.shape," y_train.shape: ",y_train.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)

#ratio of negatives/positives weights -> (0-Bkgd)/(1-Sig):
ratio = float(bkg.shape[0]/sig.shape[0])
print("ratio of majority/minority class: ", ratio)
#ratio = float(np.sum(label == 0)) / np.sum(label == 1)


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

fig2 = plt.figure(1,figsize=(15,20))

def run_model(model, alg_name, plot_index):
    # build the model on training data
    eval_set = [(X_train, y_train), (X_test, y_test)]
    #eval_set = [(X_test, y_test)]
    #model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10)
    #model.fit(X_train, y_train, eval_metric=["error"], eval_set=eval_set, early_stopping_rounds=10, verbose=True)
    #model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True, early_stopping_rounds=15,)
    model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
    #model.fit(X_train, y_train)
    #print("X_train ", X_train," y_train: ",y_train)

    # save the model to disk
    filename = 'xgboost_Classify.sav'
    pickle.dump(model, open(filename, 'wb'))

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred] #you can set a different value of probability to separate Signal evts: default:0.5

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    
    # retrieve performance metrics
    results = model.evals_result() 
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    
    # plot log loss
    fig, ax = pyplot.subplots(figsize=(12,12))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()    
    pyplot.ylabel('Log Loss')
    pyplot.title('XGBoost Log Loss')
    pyplot.show()
    
    # plot classification error
    fig, ax = pyplot.subplots(figsize=(12,12))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('XGBoost Classification Error')
    pyplot.show()
   
    # calculate the accuracy score
    accuracy =  accuracy_score(y_test, y_pred) * 100 #returns the fraction of correctly classified samples
    print("--------- alg_name: ",alg_name)
    print('Accuracy: %.3f' % accuracy)
    #print("y_test: ",y_test," y_pred: ",pd.DataFrame(y_pred))
    #print("y_test.count: ", y_test.count," pd.DataFrame(y_pred).count: ", pd.DataFrame(y_pred).count)
    predictions0 = pd.concat([y_test.reset_index(drop=True) ,pd.DataFrame(y_pred,columns=['Prediction'])], axis=1)
    predictions0.to_csv( str(alg_name) + '.csv')
    report = classification_report(y_test, y_pred) 
    print(report)

    #Calculate and Plot ROC curve
    from sklearn.metrics import roc_curve, auc
   
    #if alg_name=="XGBoost":
    model.probability = True
    probas = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label =1)# 0 is signal
    roc_auc  = auc(fpr, tpr)
 
    ax2 = fig2.add_subplot(1,1,plot_index)
    ax2.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (alg_name, roc_auc))
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc=0, fontsize='small')

    #plot prediction probabilities:
    probas_train = model.predict_proba(X_train)
    pred_train = pd.concat([y_train.reset_index(drop=True) ,pd.DataFrame(probas_train[:, 1],columns = ['probas'])], axis=1)
    probasTrain_trueSig = pred_train.loc[pred_train["newlabel"]==0]
    probasTrain_trueBkg = pred_train.loc[pred_train["newlabel"]==1]

    out_pred_prob = pd.concat([y_test.reset_index(drop=True) ,pd.DataFrame(probas[:, 1],columns = ['probas'])], axis=1)
    print(type(y_test), y_test.shape)
    print(type(probas), probas[:, 1].shape)
    print(out_pred_prob.head())
    probas_trueSig = out_pred_prob.loc[out_pred_prob["newlabel"]==0]
    probas_trueBkg = out_pred_prob.loc[out_pred_prob["newlabel"]==1]
 
    plt.figure(figsize=(15,7))
    plt.hist(probasTrain_trueBkg["probas"], bins=100, label='AtmMuons', alpha=0.5, histtype='step', log=True)
    plt.hist(probasTrain_trueSig["probas"], bins=100, label='Neutrinos', alpha=0.5, histtype='step', log=True)
    plt.hist(probas_trueBkg["probas"], bins=50, label='AtmMuons', alpha=0.7, color='r',log=True)
    plt.hist(probas_trueSig["probas"], bins=50, label='Neutrinos', alpha=0.7, color='b',log=True)
    plt.xlabel('Probability of being predicted as Neutrino', fontsize=25)
    plt.ylabel('Number of evnts', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.show() 


# ----- xgboost ------------
# install xgboost
# 'pip install xgboost' or https://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/39811079#39811079

from xgboost import XGBClassifier

#model = XGBClassifier()
#model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1.0, gamma=2, learning_rate=0.02, max_delta_step=0,
#       max_depth=5, min_child_weight=5, missing=None, n_estimators=600,
#       n_jobs=1, nthread=1, objective='binary:logistic', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#       silent=True, subsample=0.8)
#model = XGBClassifier(subsample=1.0, n_estimators=500, min_child_weight=5, max_depth=3, learning_rate= 0.15, gamma=0.5, colsample_bytree=0.7)
###Solving the imbalance:
#scale_pos_weight=(#evts in the majority class)/(#evts in the minority class)
#model = XGBClassifier(subsample=1.0, n_estimators=200, min_child_weight=4, max_depth=3, learning_rate= 0.2, gamma=0.5, colsample_bytree=1.0, scale_pos_weight=8) 
#{'subsample': 1.0, 'scale_pos_weight': 9, 'n_estimators': 600, 'min_child_weight': 1, 'max_depth': 6, 'max_delta_step': 0, 'learning_rate': 0.2, 'gamma': 1.5, 'colsample_bytree': 0.7}
#model = XGBClassifier(subsample=1.0, n_estimators=600, min_child_weight=1, max_depth=6, learning_rate= 0.2, gamma=1.5, colsample_bytree=0.7, scale_pos_weight=9, objective='binary:logistic') 
model = XGBClassifier(scale_pos_weight= 8)
#model = XGBClassifier(subsample=1.0, n_estimators=600, min_child_weight=1, max_depth=6, learning_rate= 0.2, gamma=1.5, colsample_bytree=0.7, scale_pos_weight=9, objective='binary:logistic') 
#BEST Model so far:
#model = XGBClassifier(subsample=1.0, n_estimators=500, min_child_weight=5, max_depth=3, learning_rate= 0.15, gamma=0.5, colsample_bytree=0.7)
#Good model :
#model = XGBClassifier(subsample=0.8, n_estimators=600, min_child_weight=10, max_depth=4, learning_rate= 0.05, gamma=0.5, colsample_bytree=0.6)
run_model(model, "XGBoost", 1)

'''
#----------- GradientBoostingClassifier -----------
from sklearn.ensemble import GradientBoostingClassifier

#model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
#model = GradientBoostingClassifier(learning_rate=0.1, max_depth=8, n_estimators=100)
model = GradientBoostingClassifier()
run_model(model, " GradientBoostingClassifier ", 1)
'''

#plt.show()
#fig.savefig("result.png")
fig2.savefig("imbalancedROCcurve.png")
