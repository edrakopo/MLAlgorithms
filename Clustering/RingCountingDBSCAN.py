#check: https://www.programcreek.com/python/example/103494/sklearn.cluster.DBSCAN

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d
# #############################################################################
# Generate sample data

#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                            random_state=0)

#X = StandardScaler().fit_transform(X)
#print("X.shape: ",X.shape," type(X)",type(X)," len(X) ", len(X))
#print("labels_true.shape: ",labels_true.shape," type(labels_true): ",type(labels_true))
#print(labels_true)

#df1=pd.DataFrame.from_records(X)
#df1.columns = ["hitsX","hitsY"]
##print(df1.head())
#df2 = pd.DataFrame(labels_true,columns=['NoOfClusters'])
#df_final = pd.concat([df1,df2],axis=1)
#print( df_final.head())
#df_final.to_csv("data/EventTableRings.csv", float_format = '%.4f')

#Read data
infile = "data/EventTableRings.csv"
filein = open(str(infile))
print("data in: ",filein)
df0=pd.read_csv(filein)
print(df0.head())
booldf = df0["Radius in m"].isnull()
null_columns=df0.columns[df0.isnull().any()]
nulldf = df0[df0["Radius in m"].isnull()][null_columns]
#print(nulldf)
print(nulldf.index.values)
#print(df0[140:145])
count = 0
for g, dfs in nulldf.groupby(np.arange(len(nulldf)) // 3):
    print("----- g: ",g," dfs: ",dfs)
    a = dfs.index.values
    #print("null indices : ",a)
    if g==0: 
       df_evt = df0.loc[:a[0]-1,:]
       count = a[0]+3
    else:
       #print("count: ",count)
       #print("from index: ", count , " to index: ", a[0]-1)
       df_evt = df0.loc[count:a[0]-1,:]
       count = a[0]+3
    print("df_evt.head(): ",df_evt.head())

    #drop LAPPDs: 
    #indexNames = df_evt[df_evt['Detector type']=='LAPPD'].index
    #df_evt.drop(indexNames , inplace=True)
    #print("PMTS only-df_evt.head(): ",df_evt.head())
 
    #drop some columns from the data sample
    X  = df_evt.drop(["Detector type", "Ring number", "Radius in m","Angle in rad","Height in m"], axis=1).values
    #print(X)
    print("X.shape: ",X.shape," type(X)",type(X)," len(X) ", len(X))
    #store true labels
    labels_true = df_evt["Ring number"].values
    print("labels_true.shape: ",labels_true.shape," type(labels_true): ",type(labels_true))
    print("Number of Rings: ",labels_true[0])

    # #############################################################################
    if g<5: 
       # Compute DBSCAN
       #db = DBSCAN(eps=0.3, min_samples=10).fit(X)
       #db = DBSCAN(eps=0.3, min_samples=4).fit(X)
       db = DBSCAN().fit(X)
       core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
       core_samples_mask[db.core_sample_indices_] = True #index of each hit
       labels = db.labels_ #in which cluster each hit belongs, if noise: -1
       #print("labels: ",labels)
       #print(db.core_sample_indices_)

       # Number of clusters in labels, ignoring noise if present.
       n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
       n_noise_ = list(labels).count(-1)

       print('Estimated number of clusters: %d' % n_clusters_)
       print('Estimated number of noise points: %d' % n_noise_)
       #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
       #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
       #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
       #print("Adjusted Rand Index: %0.3f"
       #      % metrics.adjusted_rand_score(labels_true, labels))
       #print("Adjusted Mutual Information: %0.3f"
       #      % metrics.adjusted_mutual_info_score(labels_true, labels))
       #print("Silhouette Coefficient: %0.3f"
       #      % metrics.silhouette_score(X, labels))

       # #############################################################################
       # Plot result
       import matplotlib.pyplot as plt

       #------ 3D plot of MC hit points: 
       fig = plt.figure()
       ax = plt.axes(projection='3d')
       ax.scatter3D(X[:,0], X[:,1],X[:,2], c=X[:,2], cmap='Greens')
       ax.set_xlabel('x')
       ax.set_ylabel('y')
       ax.set_zlabel('z')
       plt.title('True number of clusters: %d' % labels_true[0])
       #plt.show()  
       plt.savefig("plots/evt"+str(g)+".png")

       #------ 3D plot of clusters:
 #      fig = plt.figure()
       fig, ax = plt.subplots()
#       global X, geo
#       ax = fig.add_subplot(geo + 5, projection='3d', title='dbscan')
#       dbscan = cluster.DBSCAN()
#       dbscan.fit(X)
#       res = dbscan.labels_
       core = db.core_sample_indices_
#       print(repr(core))
       size = [5 if i not in core else 40 for i in range(len(X))]
       print(repr(size))
#       # Number of clusters in labels, ignoring noise if present.
#       n_clusters_ = len(set(res)) - (1 if -1 in res else 0)
#       n_noise_ = list(res).count(-1)
#       print('in dbscan - Estimated number of clusters: %d' % n_clusters_)
#       print('in dbscan - Estimated number of noise points: %d' % n_noise_)
       for n, i in enumerate(X):
           #print ("n: ",n," i: ",i)
           ax.scatter(*i[: 3], s=size[n], c='bgrcmyk'[labels[n] % 7],
#           ax.scatter(*i[: 3], c='bgrcmyk'[labels[n] % 7],
                      alpha=0.8, marker='o')
 
       ax.set_xlabel('X Label')
       ax.set_ylabel('Y Label')
       ax.set_zlabel('Z Label')
#       plt.show()
       plt.savefig("plots/3Drecoring"+str(g)+".png")

       #------2D plot of clusters - black noise hits:
       fig = plt.figure() 
       # Black removed and is used for noise instead.
       unique_labels = set(labels)
       colors = [plt.cm.Spectral(each)
                 for each in np.linspace(0, 1, len(unique_labels))]
       for k, col in zip(unique_labels, colors):
           if k == -1:
               # Black used for noise.
               col = [0, 0, 0, 1]

           class_member_mask = (labels == k)

           xy = X[class_member_mask & core_samples_mask]
           plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)

           xy = X[class_member_mask & ~core_samples_mask]
           plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

       plt.title('Estimated number of clusters: %d' % n_clusters_)
       #plt.show()
       plt.savefig("plots/recoring"+str(g)+".png")
