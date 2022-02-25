import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import pyplot

Signal_file = "anal_scripts_ROOT_CSV_py/vars_Signal.csv"
Bkgd_file = "anal_scripts_ROOT_CSV_py/vars_Bkgd.csv"

columns = ["TrLengthIT_2","zenith","ratio_doms","ratio_casc_doms","jlik","logbeta0","reco_cheredoms","ratio_pmts","logEreco","jz_f","D","max_zen_sol","ratio330","ratio6","ratio4","prefitN_1degree","prefitN_up","max_diff_sollik","redToT_cascmuon","diff_dist_mu","ratio310","ratiol", "label"]

#get signal data
sig = pd.read_csv(Signal_file, usecols=columns)
print(sig.head())
print(sig.shape)
#get Bkgd data
bkg = pd.read_csv(Bkgd_file, usecols=columns)
print(bkg.head())
print(bkg.shape)

#Merge signal and Bkgd
data0 = pd.concat([sig,bkg],axis=0)
print(data0.head())

#randomly shuffle the data
data = shuffle(data0, random_state=0) 
print("after shuffling: ", data.head())
print("data.shape: ", data.shape)

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train0, X_test0, y_train, y_test = train_test_split(data.iloc[:,:22], data.iloc[:,22], test_size = 0.3, random_state = 100)
print(X_train0.head(), " y: ", y_train.head())
print(X_test0.head(), " ytest: ", y_test.head())

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))

print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))

print("X_train.shape: ",X_train.shape," y_train.shape: ",y_train.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#fig = plt.figure(1,figsize=(15,20))
fig2 = plt.figure(2,figsize=(15,20))

def run_model(model, alg_name, plot_index):
    # build the model on training data
    model.fit(X_train, y_train)
    #    model.fit(X_train, y_train,
    #       eval_set=[(X_train, y_train), (X_test, y_test)],
    #       eval_metric='logloss',
    #       verbose=True)
    #print("X_train ", X_train," y_train: ",y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    # calculate the accuracy score
    accuracy =  accuracy_score(y_test, y_pred) * 100 #returns the fraction of correctly classified samples
    print("--------- alg_name: ",alg_name)
    print('Accuracy: %.3f' % accuracy)
    #print("y_test: ",y_test," y_pred: ",pd.DataFrame(y_pred))
    #print("y_test.count: ", y_test.count," pd.DataFrame(y_pred).count: ", pd.DataFrame(y_pred).count)
    predictions = pd.concat([y_test.reset_index(drop=True) ,pd.DataFrame(y_pred,columns=['Prediction'])], axis=1)
    #if alg_name=="XGBoost":
    #   predictions.to_csv('XGBoost_predictions.csv')
    predictions.to_csv( str(alg_name) + '.csv')
    report = classification_report(y_test, y_pred) 
    print(report)
    '''
    if alg_name=="XGBoost":    
       # retrieve performance metrics
       results = model.evals_result()
       # plot learning curves
       pyplot.plot(results['validation_0']['logloss'], label='train')
       pyplot.plot(results['validation_1']['logloss'], label='test')
       # show the legend
       pyplot.legend()
       # show the plot
       pyplot.show()
    '''

    #Calculate and Plot ROC curve
    from sklearn.metrics import roc_curve, auc
   
    #if alg_name=="XGBoost":
    model.probability = True
    probas = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label =1)# 1 is signal
    roc_auc  = auc(fpr, tpr)
 
    ax2 = fig2.add_subplot(5,1,plot_index)
    ax2.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (alg_name, roc_auc))
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc=0, fontsize='small')
#       fig2, ax2 = plt.subplots(nrows=2, ncols=1) 
#       plt.figure(2)
#       plt.plot([0, 1], [0, 1], 'k--')
#       plt.xlim([0.0, 1.0])
#       plt.ylim([0.0, 1.0])
#       plt.xlabel('False Positive Rate')
#       plt.ylabel('True Positive Rate')
#       plt.legend(loc=0, fontsize='small')
#       plt.savefig("ROCcurve_e_mulessALL.png")


#------ Test different models:
# ---- Decision Tree -----------
from sklearn import tree

#model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
#model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
model = tree.DecisionTreeClassifier(max_depth=10)
run_model(model, "Decision Tree", 1)

# ----- Random Forest ---------------
from sklearn.ensemble import RandomForestClassifier

#model = RandomForestClassifier(n_estimators=10)
model = RandomForestClassifier(n_estimators=100)
run_model(model, "Random Forest", 2)

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
model = XGBClassifier(subsample=1.0, n_estimators=600, min_child_weight=10, max_depth=5, learning_rate=0.1, gamma=0.5, colsample_bytree=1.0)
run_model(model, "XGBoost", 3)

# ---------- SGD Classifier -----------------
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

model = OneVsRestClassifier(SGDClassifier(loss="log", max_iter=1000))
run_model(model, "SGD Classifier", 4)

#----------- GradientBoostingClassifier -----------
from sklearn.ensemble import GradientBoostingClassifier

#model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=8, n_estimators=100)
run_model(model, " GradientBoostingClassifier ", 5)

#plt.show()
#fig.savefig("result.png")
fig2.savefig("ROCcurve.png")
