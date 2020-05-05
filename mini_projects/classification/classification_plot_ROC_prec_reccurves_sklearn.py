import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets as d
from sklearn import preprocessing

#create an example dataset for binary clasification (for simplicity x has two features):
X, y = d.make_classification(n_samples=10000, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, flip_y=0.2)
#print("X_data: ",X)
#print("Y_data: ",y)

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train0, X_test0, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train0))
X_test = pd.DataFrame(scaler.transform(X_test0))

y_train2 = np.array(y_train).ravel() #Return a contiguous flattened 1-d array
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape, "X_test.shape: ",X_test.shape," y_test.shape: ",y_test.shape)


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#fig = plt.figure(1,figsize=(10,10))
#fig2 = plt.figure(1,figsize=(10,10))

def run_model(model, alg_name, plot_index):
    # build the model on training data
    model.fit(X_train, y_train2)
    #print("X_train ", X_train," y_train: ",y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    # calculate the accuracy score
    accuracy =  accuracy_score(y_test, y_pred) * 100 #returns the fraction of correctly classified samples
    print("--------- alg_name: ",alg_name)
    report = classification_report(y_test, y_pred) 
    print(report)

    #Calculate and Plot ROC curve
    from sklearn.metrics import roc_curve, auc
   
    model.probability = True
    probas = model.predict_proba(X_test)
    print("probas ",probas," y_test ",y_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label =1)
    roc_auc  = auc(fpr, tpr)
 
#    ax = fig.add_subplot(2,2,plot_index)
#    ax.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (alg_name, roc_auc))
#    ax.plot([0, 1], [0, 1], 'k--')
#    ax.set_xlabel('False Positive Rate')
#    ax.set_ylabel('True Positive Rate')
#    ax.legend(loc=0, fontsize='small')

    # Plot the Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
 #   from sklearn.metrics import plot_precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_test, probas[:, 1], pos_label =1)
#    ax2 = fig2.add_subplot(2,2,plot_index)
#    ax2.plot(recall, precision, marker='.', label='model')
#    ax2.set_xlabel('Recall')
#    ax2.set_ylabel('Precision')

#------ Test different models:
from xgboost import XGBClassifier

model = XGBClassifier(subsample=1.0, n_estimators=600, min_child_weight=10, max_depth=5, learning_rate=0.1, gamma=0.5, colsample_bytree=1.0)
run_model(model, "XGBoost", 1)

# ----------- Neural network - Multi-layer Perceptron  ------------
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes= 100, activation='relu')
run_model(model, " MLP Neural network ", 2)

#----------- LogisticRegression ------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l1', tol=0.01) 
run_model(model, " LogisticRegression ", 3)

#----------- GradientBoostingClassifier -----------
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.1, max_depth=8, n_estimators=100)
run_model(model, " GradientBoostingClassifier ", 4)

#plt.show()
#fig.savefig("ROCcurve_sklearn.png")
#fig2.savefig("prec_reccurve_sklearn.png")
