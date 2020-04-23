#https://github.com/Rockyzsu/base_function/blob/master/sklearn_basic.py
from __future__ import unicode_literals
import datetime
from collections import Counter

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets


iris = datasets.load_iris()
# X_iris = iris.data[50: 100]
X_iris = iris.data
Y_iris = iris.target
geo = 231
print("X_iris: ",X_iris)
print("Y_iris: ",Y_iris)
# X_iris = np.delete(X_iris, 3, axis=1)
# X_iris /= 10.


def timeit(name=None):
    """
    @decorate
    """
    def wrapper2(func):
        def wrapper1(*args, **kargs):
            start = datetime.datetime.now()
            r = func(*args, **kargs)
            end = datetime.datetime.now()
            print('---------------')
            print('project name: %s' % name)
            print('start at: %s' % start)
            print('end at:   %s' % end)
            print('cost:     %s' % (end - start))
            print('res:      %s' % r)
#            print('err1:     %s' \)
            #      % (50 - Counter(r[: 50]).most_common()[0][1])
#            print('err2:     %s' \)
            #      % (50 - Counter(r[50: 100]).most_common()[0][1])
#            print('err3:     %s' \)
            #      % (50 - Counter(r[100: 150]).most_common()[0][1])
            print('---------------')
            return r
        return wrapper1
    return wrapper2


def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin


@timeit('target')
def target(fig):
    global X_iris, Y_iris, geo
    ax = fig.add_subplot(geo + 0, projection='3d', title='target')
    for n, i in enumerate(X_iris):
        ax.scatter(*i[: 3], c=['r', 'y', 'g'][Y_iris[n]], marker='o')
        print("n ",n," Y_iris[n] ",Y_iris[n]," type: ",type(Y_iris[n]))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return Y_iris


# kmeans
@timeit('kmeans')
def kmeans(fig):
    global X_iris, geo
    ax = fig.add_subplot(geo + 1, projection='3d', title='k-means')
    k_means = cluster.KMeans(init='random', n_clusters=3)
    k_means.fit(X_iris)
    res = k_means.labels_
    for n, i in enumerate(X_iris):
        ax.scatter(*i[: 3], c='bgrcmyk'[res[n] % 7], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return res


@timeit('mini_batch_kmeans')
def mini_batch(fig):
    global X_iris, geo
    ax = fig.add_subplot(geo + 2, projection='3d', title='mini-batch')
    mini_batch = cluster.MiniBatchKMeans(init='random', n_clusters=3)
    mini_batch.fit(X_iris)
    res = mini_batch.labels_
    for n, i in enumerate(X_iris):
        ax.scatter(*i[: 3], c='bgrcmyk'[res[n] % 7], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return res


@timeit('affinity')
def affinity(fig):
    global X_iris, geo
    ax = fig.add_subplot(geo + 3, projection='3d', title='affinity')
    affinity = cluster.AffinityPropagation(preference=-50)
    affinity.fit(X_iris)
    res = affinity.labels_
    for n, i in enumerate(X_iris):
        ax.scatter(*i[: 3], c='bgrcmyk'[res[n] % 7], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return res


@timeit('mean_shift')
def mean_shift(fig):
    global X_iris, geo
    ax = fig.add_subplot(geo + 4, projection='3d', title='mean_shift')
    bandwidth = cluster.estimate_bandwidth(X_iris, quantile=0.2, n_samples=50)
    mean_shift = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift.fit(X_iris)
    res = mean_shift.labels_
    for n, i in enumerate(X_iris):
        ax.scatter(*i[: 3], c='bgrcmyk'[res[n] % 7], marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return res


@timeit('dbscan')
def dbscan(fig):
    global X_iris, geo
    ax = fig.add_subplot(geo + 5, projection='3d', title='dbscan')
    dbscan = cluster.DBSCAN()
    dbscan.fit(X_iris)
    res = dbscan.labels_
    core = dbscan.core_sample_indices_
    print(repr(core))
    size = [5 if i not in core else 40 for i in range(len(X_iris))]
    print(repr(size))
    for n, i in enumerate(X_iris):
        ax.scatter(*i[: 3], s=size[n], c='bgrcmyk'[res[n] % 7],
                   alpha=0.8, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return res


def main():
    fig = plt.figure()
    target(fig)
    kmeans(fig)
    mini_batch(fig)
    affinity(fig)
    mean_shift(fig)
    dbscan(fig)

    plt.show()


if __name__ == '__main__':
    main()
