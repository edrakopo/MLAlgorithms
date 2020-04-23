import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing

#------- Merge .csv files -------
#filenames = ["data/electrons_uniform.csv","data/muons_uniform.csv"] 
#data = pd.concat( [ pd.read_csv(f) for f in filenames ] ) merge .csv files
#print("combined_csv: ", data)
#data_e = pd.read_csv("data/electrons_uniform.csv", names = ["Dist from vertex to PMT (m)", "Angle (rad)", "Energy (MeV)", "Expected # of p.e", "Expected hit time"])
data_e = pd.read_csv("data/electrons_uniform_x20_y20_z10_Parametric.csv", names = ["Dist from vertex to PMT (m)", "Angle (rad)", "Energy (MeV)", "Expected # of p.e", "Expected hit time"])
data_e['Particle Type'] = "electron"

#data_mu = pd.read_csv("data/muons_uniform.csv", names = ["Dist from vertex to PMT (m)", "Angle (rad)", "Energy (MeV)", "Expected # of p.e", "Expected hit time"])
data_mu = pd.read_csv("data/muons_uniform_x20_y20_z10_Parametric.csv", names = ["Dist from vertex to PMT (m)", "Angle (rad)", "Energy (MeV)", "Expected # of p.e", "Expected hit time"])
data_mu['Particle Type'] = "muon"

#print("data_e: ", data_e)
#print("data_mu: ", data_mu)
data = pd.concat([data_e,data_mu],axis=0)

##use only a slice of data:
#data_e0 = data_e[:60000]
#data_mu0 = data_mu[:60000]
#data = pd.concat([data_e0,data_mu0],axis=0)

# ------ Load data -----------
X = data.iloc[:,0:5]  # ignore first column which is row Id
y = data.iloc[:,5:6]  # Classification on the 'Species'

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


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

fig = plt.figure(1,figsize=(15,20))
fig2 = plt.figure(2,figsize=(15,20))

def run_model(model, alg_name, plot_index):
    # build the model on training data
    model.fit(X_train, y_train2)
    #print("X_train ", X_train," y_train: ",y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    # calculate the accuracy score
    accuracy =  accuracy_score(y_test, y_pred) * 100 #returns the fraction of correctly classified samples
    print("--------- alg_name: ",alg_name)
    #print("y_test: ",y_test," y_pred: ",pd.DataFrame(y_pred))
    #print("y_test.count: ", y_test.count," pd.DataFrame(y_pred).count: ", pd.DataFrame(y_pred).count)
    predictions = pd.concat([y_test.reset_index(drop=True) ,pd.DataFrame(y_pred,columns=['Prediction'])], axis=1)
    if alg_name=="XGBoost":
       predictions.to_csv('XGBoost_predictions.csv')
    report = classification_report(y_test, y_pred) 
    print(report)

    # Compare the prediction result with ground truth -- PLOTS
    #color_code = {'virginica':'red', 'setosa':'blue', 'versicolor':'green'}
    color_code = {'muon':'red', 'electron':'blue'}

    # plt.figure(plot_index)
    ax = fig.add_subplot(5,2,plot_index) #nrows, ncols, index
    colors = [color_code[x] for x in y_test.iloc[:,0]]
    #ax.scatter(X_test.iloc[:,0], X_test.iloc[:,3], color=colors, marker='.', label='Circle = Ground truth')
    ax.scatter(X_test.iloc[:,0], X_test.iloc[:,2], color=colors, marker='.', label='Circle = Ground truth')
    colors = [color_code[x] for x in y_pred]
    #ax.scatter(X_test.iloc[:, 0], X_test.iloc[:,3], color=colors, marker='o', facecolors='none', label='Dot = Prediction')
    ax.scatter(X_test.iloc[:, 0], X_test.iloc[:,2], color=colors, marker='o', facecolors='none', label='Dot = Prediction')

    #plt.axes([0.65, 0.65, 0.2, 0.2])
    ax.legend(loc="lower right")
    # manually set legend color to black
    leg = plt.gca().get_legend()
#    leg.legendHandles[0].set_color('black')
#    leg.legendHandles[1].set_color('black')
#    leg.legendHandles[1].set_facecolors('none')

    ax.set_title(alg_name + ". Accuracy: " + str(accuracy))

    #Calculate and Plot ROC curve
    from sklearn.metrics import roc_curve, auc
   
    #if alg_name=="XGBoost":
    model.probability = True
    probas = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label ="muon")#assumes muon is the positive result
    roc_auc  = auc(fpr, tpr)
 
    ax2 = fig2.add_subplot(5,2,plot_index)
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

# ------ SVM Classifier ----------------
from sklearn.svm import SVC
model = SVC(probability=True)
run_model(model, "SVM Classifier", 4)

# -------- Nearest Neighbors ----------
from sklearn import neighbors
#model = neighbors.KNeighborsClassifier()
model = neighbors.KNeighborsClassifier(n_neighbors=10)
run_model(model, "Nearest Neighbors Classifier", 5)

# ---------- SGD Classifier -----------------
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

model = OneVsRestClassifier(SGDClassifier(loss="log", max_iter=1000))
run_model(model, "SGD Classifier", 6)

# --------- Gaussian Naive Bayes ---------
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
run_model(model, "Gaussian Naive Bayes", 7)

# ----------- Neural network - Multi-layer Perceptron  ------------
from sklearn.neural_network import MLPClassifier

#model = MLPClassifier()
#model = MLPClassifier(activation="tanh")
model = MLPClassifier(hidden_layer_sizes= 100, activation='relu')
run_model(model, " MLP Neural network ", 8)

#----------- LogisticRegression ------------
from sklearn.linear_model import LogisticRegression

#model = LogisticRegression(solver='liblinear', multi_class='ovr')
model = LogisticRegression(penalty='l1', tol=0.01) 
run_model(model, " LogisticRegression ", 9)

#----------- LinearDiscriminantAnalysis -----------
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#model = LinearDiscriminantAnalysis()
#run_model(model, " LinearDiscriminantAnalysis ", 10)

#----------- GradientBoostingClassifier -----------
from sklearn.ensemble import GradientBoostingClassifier

#model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=8, n_estimators=100)
run_model(model, " GradientBoostingClassifier ", 10)

#plt.show()
fig.savefig("result_e_mulessALL_oldCSV.png")
fig2.savefig("ROCcurve_e_mulessALL_oldCSV.png")
#plt.savefig("result_e_mulessALL.png")
#plt.savefig("result_e_muALL.png")
#plt.savefig("result_e_mu.png")
