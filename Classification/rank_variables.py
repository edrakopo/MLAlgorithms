import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.feature_selection import RFE

#------- Merge .csv files -------
data_e = pd.read_csv("data/pdf_electron_Parametric_single.csv", header = None)
data_e[11] = "electron"
data_mu = pd.read_csv("data/pdf_muon_Parametric_single.csv", header = None)
data_mu[11] = "muon"

data = pd.concat([data_e,data_mu],axis=0)

# ------ Load data -----------
X = data.iloc[:,0:11]  # ignore first column which is row Id
y = data.iloc[:,11:12]  # Classification on the 'Species'

print("X_data: ",X)
print("Y_data: ",y)

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

#------Feature Importance with ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
forest.fit(X_train,y_train2)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print(importances)
print(indices)

feature_labels=["Nhits","totalQ","averageT","angBary","angRMS","angVar","angSkew","angKurt","distVert","distHor","energy"]
feature_labels_sorted = [0] * 11

#print the feature ranking
print("ExtraTreesClassifier: ")
print("Feature ranking: ")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    feature_labels_sorted[f] = feature_labels[indices[f]]

#plot the feature importances of the forest
plt.figure()
plt.title("Feature importances - ExtraTreesClassifier")
plt.bar(range(X_train.shape[1]), importances[indices],color="b",yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=7)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance")
plt.savefig("FeatureImportances_ExtraTreesClassifier.pdf")


#-----Univariate Selection-------------------------------
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

kBest = SelectKBest(score_func=f_classif, k=5)
fit_kBest = kBest.fit(X_train,y_train2)
np.set_printoptions(precision=3)
print("Univariate Selection:")
print(fit_kBest.scores_)
#selected_features=fit_kBest.transform(X_train)
#print(selected_features[0:6,:])
indices=np.argsort(fit_kBest.scores_)[::-1]
print(indices)
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], fit_kBest.scores_[indices[f]]))
    feature_labels_sorted[f] = feature_labels[indices[f]]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - KBest Univariate Selection")
plt.bar(range(X_train.shape[1]), fit_kBest.scores_[indices],color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=7)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance")
plt.savefig("FeatureImportances_UnivariateSelection.pdf")

#------Principal Component Analysis------------------------
#from sklearn.decomposition import PCA
# not really sure how to interprete reduced data --> omit for now

#pca = PCA(n_components=5)
#fit_pca = pca.fit(X_train)
#print("Principal Component Analysis:")
#print(("Explained Variance: %s") % fit_pca.explained_variance_ratio_)
#print(fit_pca.components_)


#-----Recursive Feature Elimination (RFE)-----------------
#------ Test the ranking of variables for different models:
# ---- Decision Tree -----------
from sklearn import tree

model = tree.DecisionTreeClassifier(max_depth=10)
rfe_tree = RFE(model,1)
rfe_tree.fit(X_train,y_train2)

print("Decision Tree:")
print(rfe_tree.support_)
print(rfe_tree.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_tree.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)
importancesAU = [11,10,9,8,7,6,5,4,3,2,1]

plt.figure()
plt.title("Feature importances - Decision Tree")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=7)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
plt.savefig("FeatureImportances_DecisionTree.pdf")



# ----- Random Forest ---------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
rfe_random = RFE(model,1)
rfe_random.fit(X_train,y_train2)

print("Random Forest:")
print(rfe_random.support_)
print(rfe_random.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_random.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - Random Forest")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=7)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
#plt.show()
plt.savefig("FeatureImportances_RandomForest.pdf")

# ----- xgboost ------------
from xgboost import XGBClassifier

model = XGBClassifier(subsample=0.6, n_estimators=100, min_child_weight=5, max_depth=4, learning_rate=0.15, gamma=0.5, colsample_bytree=1.0)
rfe_xgb = RFE(model,1)
rfe_xgb.fit(X_train,y_train2)

print("XGB classifier:")
print(rfe_xgb.support_)
print(rfe_xgb.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_xgb.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - XGB")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=7)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
plt.savefig("FeatureImportances_XGB.pdf")

# ------ SVM Classifier ----------------
#gives error message: The classifier does not expose "coef_" or "feature_importances_" attributes (if not specifying kernel="linear")
from sklearn.svm import SVC

model = SVC(probability=True, kernel="linear")
rfe_svc = RFE(model,1)
rfe_svc.fit(X_train,y_train2)

print("SVC classifier:")
print(rfe_svc.support_)
print(rfe_svc.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_svc.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - SVM")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=7)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
plt.savefig("FeatureImportances_SVM.pdf")

# -------- Nearest Neighbors ----------
#gives error message: The classifier does not expose "coef_" or "feature_importances_" attributes
#from sklearn import neighbors

#model = neighbors.KNeighborsClassifier(n_neighbors=10)
#rfe_neighbors = RFE(model,1)
#rfe_neighbors.fit(X_train,y_train2)

#print("K Nearest Neighbors classifier:")
#print(rfe_neighbors.support_)
#print(rfe_neighbors.ranking_)

# ----------- Neural network - Multi-layer Perceptron  ------------
#gives error message: The classifier does not expose "coef_" or "feature_importances_" attributes
#from sklearn.neural_network import MLPClassifier

#model = MLPClassifier(hidden_layer_sizes= 100, activation='relu')
#rfe_MLP = RFE(model,1)
#rfe_MLP.fit(X_train,y_train2)

#print("MLP classifier:")
#print(rfe_MLP.support_)
#print(rfe_MLP.ranking_)

#----------- GradientBoostingClassifier -----------
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, n_estimators=100)
rfe_gradientboosting = RFE(model,1)
rfe_gradientboosting.fit(X_train,y_train2)

print("Gradient Boosting classifier:")
print(rfe_gradientboosting.support_)
print(rfe_gradientboosting.ranking_)

for f in range(X_train.shape[1]):
    index = np.where(rfe_gradientboosting.ranking_==f+1)[0][0]
    print(index)
    feature_labels_sorted[f] = feature_labels[index]

print(feature_labels_sorted)

plt.figure()
plt.title("Feature importances - Gradient Boosting")
plt.bar(range(X_train.shape[1]), importancesAU,color="b", align="center")
plt.xticks(range(X_train.shape[1]), feature_labels_sorted, rotation=45, fontsize=7)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel("Importance [A.U.]")
plt.savefig("FeatureImportances_GradientBoosting.pdf")

