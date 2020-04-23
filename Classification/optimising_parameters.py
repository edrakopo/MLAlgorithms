import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#------- Merge .csv files -------
#filenames = ["data/electrons_uniform.csv","data/muons_uniform.csv"] 
#data = pd.concat( [ pd.read_csv(f) for f in filenames ] ) merge .csv files
#print("combined_csv: ", data)
#data_e = pd.read_csv("data/electrons_uniform.csv", names = ["Dist from vertex to PMT (m)", "Angle (rad)", "Energy (MeV)", "Expected # of p.e", "Expected hit time"])
data_e = pd.read_csv("data/pdf_electron_Parametric_single.csv", header = None)
data_e[8] = "electron"

#data_mu = pd.read_csv("data/muons_uniform.csv", names = ["Dist from vertex to PMT (m)", "Angle (rad)", "Energy (MeV)", "Expected # of p.e", "Expected hit time"])
data_mu = pd.read_csv("data/pdf_muon_Parametric_single.csv", header = None)
data_mu[8] = "muon"

#print("data_e: ", data_e)
#print("data_mu: ", data_mu)
data = pd.concat([data_e,data_mu],axis=0)

##use only a slice of data:
#data_e0 = data_e[:60000]
#data_mu0 = data_mu[:60000]
#data = pd.concat([data_e0,data_mu0],axis=0)

# ------ Load data -----------
X = data.iloc[:,0:8]  # ignore first column which is row Id
y = data.iloc[:,8:9]  # Classification on the 'Species'

#print("X_data: ",X)
#print("Y_data: ",y)

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))

print("type(X_train) ",type(X_train)," type(X_train0): ",type(X_train0))
#print("X_train0: ",X_train0)
#print("X_train: ",X_train)

#X_train2 = np.array(X_train)  # ignore first column which is row Id
y_train2 = np.array(y_train).ravel() #Return a contiguous flattened 1-d array
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)


# ------ optimise model ------ 
def optimise_model(model, alg_name, search_method, params):
    folds = 3
    param_comb = 5
    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    #random search is faster than GridSearch but GridSearch is more detailed
    #(see https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost for details): 
    if search_method=="random":
       #---- RandomizedSearchCV ----
       random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train2), verbose=3, random_state=1001 )
    if search_method=="grid":
       #---- GridSearchCV ----
       random_search = GridSearchCV(model, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, y_train2), verbose=3)

    random_search.fit(X_train, y_train2)

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
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "n_estimators":[10, 100, 200, 500, 600],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

model = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
optimise_model(model, "XGBoost", "random", params)

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


