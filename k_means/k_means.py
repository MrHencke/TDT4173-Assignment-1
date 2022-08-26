import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, k=2):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.k = k
        self.data = pd.DataFrame(columns=["x0", "x1", "cluster"])
        self.centroids = pd.DataFrame(columns=["x0", "x1"])
        self.changes = True
        self.iterations = 0

    def update_centroids(self):
        # TODO: Shorten this if possible, do assignment once, possibly with lambda?
        for i in range(self.k):
            self.centroids.loc[i]["x0"] = self.data.loc[self.data["cluster"] == i]["x0"].mean(
            )
            self.centroids.loc[i]["x1"] = self.data.loc[self.data["cluster"] == i]["x1"].mean(
            )

    def fit(self, X: pd.DataFrame):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # Initialize random centroids
        for _ in range(self.k):
            self.centroids = self.centroids.append(
                X.sample(replace=False), ignore_index=True)

        # Copy dataset into internal data, set cluster to -1 (unassigned)
        self.data = self.data.append(X, ignore_index=True).assign(
            cluster=-1).astype({'cluster': 'int32'})

        # Perform update step until convergence
        while self.changes:
            self.changes = False
            self.data["cluster"] = self.data.apply(
                lambda row: self.update_point_cluster(row), 1)
            self.update_centroids()
            self.iterations += 1

        print(f"Fitting required {self.iterations} iterations")

    def update_point_cluster(self, row):
        cluster = min(range(self.k),
                      key=lambda i: euclidean_distance(
            self.get_centroid(i).values, row.iloc[0:2].values))
        if row["cluster"] != cluster:
            self.changes = True
            return cluster
        return row["cluster"]

    def predict(self, X: pd.DataFrame):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        result = np.empty(len(X), int)
        for index, row in X.iterrows():
            result[index] = min(
                range(self.k), key=lambda i: euclidean_distance(self.centroids.loc[i], row))
        return result

    def get_centroid(self, k: int):
        """
        Returns centroid k
        """
        return self.centroids.iloc[k]

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return np.array([self.get_centroid(i) for i in range(self.k)])


# --- Some utility functions

def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion


def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of
            n-dimensional points

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points

    Args:
        x (array<m,d>): float tensor with pairs of
            n-dimensional points.
        y (array<n,d>): float tensor with pairs of
            n-dimensional points. Uses y=x if y is not given.

    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance

    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))
