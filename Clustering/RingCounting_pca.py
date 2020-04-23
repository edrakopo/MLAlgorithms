"""
=======================================================
Comparison of LDA and PCA 2D projection of Iris dataset
=======================================================

The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.

Principal Component Analysis (PCA) applied to this data identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that
account for the most variance *between classes*. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.
"""
print(__doc__)

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#target_names = iris.target_names

#Select verbosity! 
verbose=0

#--- Read data from RingEvents.csv
infile = "data/RingEvents.csv"
filein = open(str(infile))
print("data in: ",filein)
df0=pd.read_csv(filein, names=["Detector type","Radius in m","Angle in rad","Height in m","x in m","y in m","z in m","Ring number"])
print(df0.head())
#print(df0[210:220])
print("index vals: ",df0.index.values," Columns names: ",df0.columns.values)

#--- group per event: 
index_evt = df0.index[df0['Detector type'] == 'Event number'].tolist()
if verbose==1:
   print(len(index_evt), " index_evt: ",index_evt)
for i in range(len(index_evt)):
    if verbose==1:
       print("event:", i ," index:",index_evt[i])
    if index_evt[i]==index_evt[-1]:
       dfs = df0[index_evt[i]:]
    else:
       dfs = df0[index_evt[i]:index_evt[i+1]]   
    #print("dfs: ",dfs)
    #--- drop these lines:
    skip_lines = dfs.index[dfs['Detector type'] == 'Event number'].tolist() + dfs.index[dfs['Detector type'] == 'Detector type'].tolist()
    #print("skip lines: ",dfs.index[dfs['Detector type'] == 'Event number'].tolist())
    df_evt = dfs.drop(skip_lines)
    #print("df_evt: ",df_evt)
    
    #drop some columns from the data sample
    #X  = df_evt.drop(["Detector type", "Ring number", "Radius in m","Angle in rad","Height in m"], axis=1).values
    #X0  = df_evt.drop(["Detector type", "Ring number", "Radius in m","Angle in rad","Height in m"], axis=1)
    X0  = df_evt.drop(["Detector type", "Ring number"], axis=1)
    X0['Radius in m'] = X0['Radius in m'].astype('float64')
    X0['Angle in rad'] = X0['Angle in rad'].astype('float64')
    X0['Height in m'] = X0['Height in m'].astype('float64')
    X0['x in m'] = X0['x in m'].astype('float64')
    X0['y in m'] = X0['y in m'].astype('float64')
    X0['z in m'] = X0['z in m'].astype('float64')
    X = X0.values
    #print("Columns names @ X: ",X0.columns.values)
    #if i==1:
    #   print(type(X[0])," ",X)
    if verbose==1: 
       print("X.shape: ",X.shape," type(X)",type(X)," len(X) ", len(X))
    #store true labels
    labels_true = df_evt["Ring number"].values
    nrings = max(labels_true)
    if verbose==1: 
       print("labels_true.shape: ",labels_true.shape," type(labels_true): ",type(labels_true))
       print("______________ New Event:", i ," index: ",index_evt[i]," True Number of Rings: ", nrings,"______________")
    y = labels_true.astype('int')
    target_names = df_evt["Ring number"]

pca = PCA()#n_components=2)
X_r = pca.fit(X).transform(X)
print("pca.components_: ",pca.components_)
print("pca.explained_variance_: ",pca.explained_variance_)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
