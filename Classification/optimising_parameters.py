import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.utils import shuffle

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
print("X_train.shape: ",X_train.shape," y_train.shape: ",y_train.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)
 
#ratio of negatives/positives weights -> (0-Bkgd)/(1-Sig):
ratio = float(bkg.shape[0]/sig.shape[0])
print("ratio of negatives(0-Bkgd)/positives(1-Sig)", ratio)

# ------ optimise model ------ 
def optimise_model(model, alg_name, search_method, params):
    folds = 3
    param_comb = 5
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    #random search is faster than GridSearch but GridSearch is more detailed
    #(see https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost for details): 
    if search_method=="random":
       #---- RandomizedSearchCV ----
       random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train), verbose=3, random_state=1001 )
      #scoring is None: using the estimator’s score method
      #random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, n_jobs=4, cv=skf.split(X_train, y_train), verbose=3, random_state=1001, return_train_score=False)
    if search_method=="grid":
       #---- GridSearchCV ---- 
       #scoring is None: using the estimator’s score method
       #random_search = GridSearchCV(model, param_grid=params, n_jobs=4, cv=skf.split(X_train, y_train), verbose=1,return_train_score=False)
       random_search = GridSearchCV(model, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train), verbose=3)

    #eval_set = [(X_train, y_train), (X_test, y_test)]
    eval_set = [(X_test, y_test)]
    #random_search.fit(X_train, y_train, eval_set=eval_set)
    random_search.fit(X_train, y_train, eval_metric=["error",], eval_set=eval_set) #using the estimator’s score method 
    #random_search.fit(X_train, y_train)

    print("------ Algorithm: " + str(alg_name))
    #print('\n All results:')
    #print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    #results = pd.DataFrame(random_search.cv_results_)
    #results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
    print("----------------------------------------------")

#------ Test different models:
# ----- xgboost ------------
from xgboost import XGBClassifier

# A parameter grid for XGBoost
params = {
        "learning_rate": [ 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "n_estimators":[500, 600, 620, 650, 670, 700, 800],
        'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'gamma': [0.5, 1, 1.5, 2, 3, 4, 5],
        'subsample': [0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'max_delta_step': [0, 1, 2, 3], 
        'max_depth': [3, 4, 5, 6]
        }

model = XGBClassifier(scale_pos_weight=ratio, learning_rate=0.02, n_estimators=600, objective='binary:logistic',nthread=2)
optimise_model(model, "XGBoost", "random", params)
#optimise_model(model, "XGBoost", "grid", params)
'''
# ----------- Neural network - Multi-layer Perceptron  ------------
from sklearn.neural_network import MLPClassifier

params = {
        "hidden_layer_sizes": [10, 50, 100],
        "activation": ['relu', 'tanh']
        } 
model = MLPClassifier()
optimise_model(model, " MLP Neural network ", "random", params)

#----------- GradientBoostingClassifier -----------
from sklearn.ensemble import GradientBoostingClassifier

params = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "max_depth":[3, 5, 8, 10],
    "n_estimators":[10, 100, 200, 500, 600]
    } 

model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
optimise_model(model, " GradientBoostingClassifier ", "random", params)
'''

