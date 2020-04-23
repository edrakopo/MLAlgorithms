import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = datasets.load_iris()
data = PCA(n_components=2).fit_transform(iris.data)
labels = KMeans(n_clusters=3).fit_predict(data)
for label, color in zip(range(3), ['r', 'g', 'b']):
    plt.scatter(data[labels == label, 0], data[labels == label, 1], c=color)
plt.show()

