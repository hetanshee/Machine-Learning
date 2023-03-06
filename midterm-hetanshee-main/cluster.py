"""
Author:
Date:
Description:
"""

import collections
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


######################################################################
# classes
######################################################################

class Point(object):

    def __init__(self, name, label, attrs):
        """
        A data point.

        Attributes
        --------------------
            name  -- string, name
            label -- string, label
            attrs -- numpy array, features
        """

        self.name = name
        self.label = label
        self.attrs = attrs

    # ============================================================
    # utilities
    # ============================================================

    def distance(self, other):
        """
        Return Euclidean distance of this point with other point.

        Parameters
        --------------------
            other -- Point, point to which we are measuring distance

        Returns
        --------------------
            dist  -- float, Euclidean distance
        """
        # Euclidean distance metric
        return np.linalg.norm(self.attrs - other.attrs)

    def __str__(self):
        """
        Return string representation.
        """
        return '%s: (%s, %s)' % (self.name, str(self.attrs), self.label)


class Cluster(object):

    def __init__(self, points):
        """
        A cluster (set of points).

        Attributes
        --------------------
            points -- list of Points, cluster elements
        """
        self.points = points

    def __str__(self):
        """
        Return string representation.
        """
        s = ''
        for point in self.points:
            s += str(point)
        return s

    # ============================================================
    # utilities
    # ============================================================

    def purity(self):
        """
        Compute cluster purity.

        Returns
        --------------------
            n           -- int, number of points in this cluster
            num_correct -- int, number of points in this cluster
                                with label equal to most common label in cluster
        """
        labels = []
        for p in self.points:
            labels.append(p.label)

        cluster_label, count = stats.mode(labels)
        return len(labels), np.float64(count)

    def centroid(self):
        """
        Compute centroid of this cluster.

        Returns
        --------------------
            centroid -- Point, centroid of cluster
        """

        ### ========== TODO: START ========== ###
        # set the centroid label to any value (e.g. the most common label in this cluster)
        X = []
        labels = []
        for p in self.points:
            X.append(p.attrs)
            labels.append(p.label)

        label, count = stats.mode(labels)
        centroid = Point(str(np.mean(X, 0)), label, np.mean(X, 0))
        return centroid
        ### ========== TODO: END ========== ###

    def medoid(self):
        """
        Compute medoid of this cluster, that is, the point in this cluster
        that is closest to all other points in this cluster.

        Returns
        --------------------
            medoid -- Point, medoid of this cluster
        """

        ### ========== TODO: START ========== ###
        minimum = float('inf')
        for p in self.points:
            distance = 0
            for i in self.points:
                distance += p.distance(i)

            if minimum > distance:
                minimum = distance
                medoid = p

        return medoid
        ### ========== TODO: END ========== ###

    def equivalent(self, other):
        """
        Determine whether this cluster is equivalent to other cluster.
        Two clusters are equivalent if they contain the same set of points
        (not the same actual Point objects but the same geometric locations).

        Parameters
        --------------------
            other -- Cluster, cluster to which we are comparing this cluster

        Returns
        --------------------
            flag  -- bool, True if both clusters are equivalent or False otherwise
        """

        if len(self.points) != len(other.points):
            return False

        matched = []
        for point1 in self.points:
            for point2 in other.points:
                if point1.distance(point2) == 0 and point2 not in matched:
                    matched.append(point2)
        return len(matched) == len(self.points)


class ClusterSet(object):

    def __init__(self):
        """
        A cluster set (set of clusters).

        Parameters
        --------------------
            members -- list of Clusters, clusters that make up this set
        """
        self.members = []

    # ============================================================
    # utilities
    # ============================================================

    def centroids(self):
        """
        Return centroids of each cluster in this cluster set.

        Returns
        --------------------
            centroids -- list of Points, centroids of each cluster in this cluster set
        """

        ### ========== TODO: START ========== ###
        centroids = []
        for c in self.members:
            centroids.append(c.centroid())
        return centroids
        ### ========== TODO: END ========== ###

    def medoids(self):
        """
        Return medoids of each cluster in this cluster set.

        Returns
        --------------------
            medoids -- list of Points, medoids of each cluster in this cluster set
        """

        ### ========== TODO: START ========== ###
        medoids = []
        for c in self.members:
            medoids.append(c.medoid())
        return medoids
        ### ========== TODO: END ========== ###

    def score(self):
        """
        Compute average purity across clusters in this cluster set.

        Returns
        --------------------
            score -- float, average purity
        """

        total_correct = 0
        total = 0
        for c in self.members:
            n, n_correct = c.purity()
            total += n
            total_correct += n_correct
        return total_correct / float(total)

    def equivalent(self, other):
        """
        Determine whether this cluster set is equivalent to other cluster set.
        Two cluster sets are equivalent if they contain the same set of clusters
        (as computed by Cluster.equivalent(...)).

        Parameters
        --------------------
            other -- ClusterSet, cluster set to which we are comparing this cluster set

        Returns
        --------------------
            flag  -- bool, True if both cluster sets are equivalent or False otherwise
        """

        if len(self.members) != len(other.members):
            return False

        matched = []
        for cluster1 in self.members:
            for cluster2 in other.members:
                if cluster1.equivalent(cluster2) and cluster2 not in matched:
                    matched.append(cluster2)
        return len(matched) == len(self.members)

    # ============================================================
    # manipulation
    # ============================================================

    def add(self, cluster):
        """
        Add cluster to this cluster set (only if it does not already exist).

        If the cluster is already in this cluster set, raise a ValueError.

        Parameters
        --------------------
            cluster -- Cluster, cluster to add
        """

        if cluster in self.members:
            raise ValueError

        self.members.append(cluster)


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k):
    """
    Randomly select k unique elements from points to be initial cluster centers.

    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO: START ========== ###
    # hint: use np.random.choice

    return np.random.choice(points, k, replace=False)
    ### ========== TODO: END ========== ###


def cheat_init(points):
    """
    Initialize clusters by cheating!

    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.

    Parameters
    --------------------
        points         -- list of Points, dataset

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO: START ========== ###
    initial_points = []
    labels = collections.defaultdict(list)
    for p in points:
        labels[p.label].append(p)

    for l in labels:
        initial_points.append(Cluster(labels[l]).medoid())

    return initial_points
    ### ========== TODO: END ========== ###


def kAverages(points, k, average=ClusterSet.centroids, init='random', plot=False):
    """
    Cluster points into k clusters using variations of k-means algorithm.

    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        init    -- string, method of initialization
                   allowable:
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm

    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """

    ### ========== TODO: START ========== ###
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to
    #       create new Cluster objects and a new ClusterSet object. Then
    #       update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).

    if init == "random":
        centroids = random_init(points, k)

    elif init == "cheat":
        centroids = cheat_init(points)

    else:
        raise ValueError

    k_clusters = ClusterSet()
    count = 0
    old = ClusterSet()
    for c in centroids:
        k_clusters.add(Cluster([c]))

    centroids = average(k_clusters)
    while not k_clusters.equivalent(old):
        assignment = collections.defaultdict(list)
        for p in points:
            min_dist = float('inf')
            best_c = None
            for c in centroids:
                dist = p.distance(c)
                if dist < min_dist:
                    min_dist = dist
                    best_c = c

            assignment[best_c].append(p)

        old = k_clusters
        k_clusters = ClusterSet()
        for c in assignment:
            k_clusters.add(Cluster(assignment[c]))

        centroids = average(k_clusters)
        if plot:
            plot_clusters(k_clusters, 'Iteration' + str(count), average)
        count += 1

    return k_clusters
    ### ========== TODO: END ========== ###

def kMeans(points, k, init='random', plot=False):

    return kAverages(points, k, ClusterSet.centroids, init, plot)


def kMedoids(points, k, init='random', plot=False):
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO: START ========== ###
    return kAverages(points, k, ClusterSet.medoids, init, plot)
    ### ========== TODO: END ========== ###


def plot_clusters(clusters, title, average):
    """
    Plot clusters along with average points of each cluster.
    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """

    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids:
        coord = c.attrs
        plt.plot(coord[0], coord[1], 'ok', markersize=12)
    for cluster in clusters.members:
        label += 1
        colors[label] = np.random.rand(3, )
        for point in cluster.points:
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()