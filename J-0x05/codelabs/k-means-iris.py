#!/usr/bin/env python

"""
K-Means Clustering

Apprentissage non-supervisé avant 2006.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

"""
Combien de centroids?

Un détails peu discuté au début est le nombre de centroids à utiliser.

    Parameter sweep
    Overfitting / underfitting
    Elbow method

"""

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
X = np.hstack((cluster1, cluster2)).T
# Et si on faisait...?
# X = np.vstack((cluster1, cluster2)).T

K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(
            cdist(X, kmeans.cluster_centers_, 'euclidean'), 
            axis=1)) / X.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()

"""
=========================================================
K-means Clustering
=========================================================

First we'll show the ground truth: the correctly labeled
points.

By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.

Noter que les plots 3D sont manoeuvrables !
"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target
fignum = 1
fig_xy = (200, 160)

# Plot the ground truth
fig = plt.figure(fignum, figsize=fig_xy)
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()

# Let's not repeat ourselves.
def plot_clusters(name, est):
    fig = plt.figure(fignum, figsize=fig_xy)
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    plt.show()

"""First let's plot what a K-means algorithm would yield
using three clusters."""
print(__doc__)
name = 'k_means_iris_3'
est = KMeans(n_clusters=3)
plot_clusters(name, est)

"""Here we'll try eight clusters."""
print(__doc__)
name = 'k_means_iris_8'
est = KMeans(n_clusters=8)
plot_clusters(name, est)

"""Finally we explore the effect of a bad
initialization on classification: not much, it turns out.
"""
name = 'k_means_iris_bad_init'
est = KMeans(n_clusters=3, n_init=1, init='random')
plot_clusters(name, est)
