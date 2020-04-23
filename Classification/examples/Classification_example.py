import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets

# ------ Load iris -----------
iris = pd.read_csv("iris.csv") #load the dataset
#iris = datasets.load_iris()
#print(iris[0:2])
#X = iris.data
#y = iris.target

#print("X: ", X)
#print("y: ", y)

X = iris.iloc[:,0:4]  # ignore first column which is row Id
y = iris.iloc[:,4:5]  # Classification on the 'Species'

print("X_iris: ",X)
print("Y_iris: ",y)

# build train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 100)

#X_train2 = np.array(X_train)  # ignore first column which is row Id
y_train2 = np.array(y_train).ravel()
print("X_train.shape: ",X_train.shape," y_train2.shape: ",y_train2.shape)


from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

fig = plt.figure(figsize=(15,20))

def run_model(model, alg_name, plot_index):
    # build the model on training data
    model.fit(X_train, y_train2)
    #print("X_train ", X_train," y_train: ",y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    # calculate the accuracy score
    accuracy =  accuracy_score(y_test, y_pred) * 100

    # Compare the prediction result with ground truth -- PLOTS
    color_code = {'virginica':'red', 'setosa':'blue', 'versicolor':'green'}

    # plt.figure(plot_index)
    ax = fig.add_subplot(4,2,plot_index) #nrows, ncols, index
    colors = [color_code[x] for x in y_test.iloc[:,0]]
    ax.scatter(X_test.iloc[:,0], X_test.iloc[:,3], color=colors, marker='.', label='Circle = Ground truth')
    #ax.scatter(X_test.iloc[:,1], X_test.iloc[:,2], color=colors, marker='.', label='Circle = Ground truth')
    colors = [color_code[x] for x in y_pred]
    ax.scatter(X_test.iloc[:, 0], X_test.iloc[:,3], color=colors, marker='o', facecolors='none', label='Dot = Prediction')
    #ax.scatter(X_test.iloc[:, 1], X_test.iloc[:,2], color=colors, marker='o', facecolors='none', label='Dot = Prediction')

    #plt.axes([0.65, 0.65, 0.2, 0.2])
    ax.legend(loc="lower right")
    # manually set legend color to black
    leg = plt.gca().get_legend()
    leg.legendHandles[0].set_color('black')
    leg.legendHandles[1].set_color('black')
    leg.legendHandles[1].set_facecolors('none')

    ax.set_title(alg_name + ". Accuracy: " + str(accuracy))

# ---- Decision Tree -----------
from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
run_model(model, "Decision Tree", 1)

# ----- Random Forest ---------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)
run_model(model, "Random Forest", 2)

# ----- xgboost ------------
# install xgboost
# 'pip install xgboost' or https://stackoverflow.com/questions/33749735/how-to-install-xgboost-package-in-python-windows-platform/39811079#39811079

from xgboost import XGBClassifier

model = XGBClassifier()
run_model(model, "XGBoost", 3)

# ------ SVM Classifier ----------------
from sklearn.svm import SVC
model = SVC()
run_model(model, "SVM Classifier", 4)

# -------- Nearest Neighbors ----------
from sklearn import neighbors
model = neighbors.KNeighborsClassifier()
run_model(model, "Nearest Neighbors Classifier", 5)

# ---------- SGD Classifier -----------------
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

model = OneVsRestClassifier(SGDClassifier())
run_model(model, "SGD Classifier", 6)

# --------- Gaussian Naive Bayes ---------
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
run_model(model, "Gaussian Naive Bayes", 7)

# ----------- Neural network - Multi-layer Perceptron  ------------
from sklearn.neural_network import MLPClassifier

model = MLPClassifier()
run_model(model, " MLP Neural network ", 8)


#plt.show()
plt.savefig("result.png")
