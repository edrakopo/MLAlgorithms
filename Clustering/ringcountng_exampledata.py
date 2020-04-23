import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d
from sklearn import cluster, datasets

from sklearn.datasets import make_moons
from matplotlib import pyplot
from pandas import DataFrame
# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                            random_state=0)

#### Make a large circle containing a smaller circle in 2d: 
#n_samples = 1500
#noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5,
#                                      noise=.05)
#datasets = [(noisy_circles, {'n_clusters': 2})]

#default_base = {'n_neighbors': 10,
#                'n_clusters': 5}
#for i_dataset, (dataset, algo_params) in enumerate(datasets):
#    # update parameters with dataset-specific values
#    params = default_base.copy()
#    params.update(algo_params)
#    X, labels_true = dataset
#    # normalize dataset for easier parameter selection
#    X = StandardScaler().fit_transform(X)

#### Make moons:
# generate 2d classification dataset
X, labels_true = make_moons(n_samples=100, noise=0.1)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=labels_true))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

#############

#print("X before scaler: ", X)
#X = StandardScaler().fit_transform(X)
print("X",X)
print("X.shape: ",X.shape," type(X)",type(X)," len(X) ", len(X))
print("labels_true.shape: ",labels_true.shape," type(labels_true): ",type(labels_true))
print(labels_true)

df1=pd.DataFrame.from_records(X)
df1.columns = ["hitsX","hitsY"]
#print(df1.head())
df2 = pd.DataFrame(labels_true,columns=['NoOfClusters'])
df_final = pd.concat([df1,df2],axis=1)
print( df_final.head())
df_final.to_csv("cluster_rings_data.csv", float_format = '%.4f')

# #############################################################################
# Compute DBSCAN
#db = DBSCAN(eps=0.3, min_samples=10).fit(X)
db = DBSCAN(eps=0.4, min_samples=10).fit(X)
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
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

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
plt.show()

#2D plot of points: 
fig = plt.figure()
ax = plt.axes()
ax.scatter(X[:,0], X[:,1])
plt.show()   
