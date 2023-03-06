"""
Author:
Date:
Description:
"""

import numpy as np
import matplotlib.pyplot as plt

import util
# TODO: change cluster_5350 to cluster if you do the extra credit
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y):
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in range(n):
        images[y[i]].append(X[i,:])

    points = []
    for face in images:
        count = 0
        for im in images[face]:
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def generate_points_2d(N, seed=1234):
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma):
        label += 1
        for i in range(N):
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# main
######################################################################

def main():
    ### ========== TODO: START ========== ###
    # part 1: explore LFW data set
    # TODO: display samples images and "average" image
    # TODO: display top 12 eigenfaces
    # TODO: try lower-dimensional representations

    X, y = util.get_lfw_data()
    n, d = X.shape
    print("5 Sample Images:")
    for i in range(5):
        util.show_image(X[i])

    average_image = []
    print("Average of images:")
    for i in range(d):
        average_image.append(np.mean(X[:, i], axis=0))
    util.show_image(np.array(average_image))


    print("Top 12 Eigenfaces:")
    U, mu = util.PCA(X)
    util.plot_gallery([util.vec_to_image(U[:, i]) for i in range(12)])

    print("Applying PCA and Reconstructing:")
    for l in [1, 10, 50, 100, 500, 1288]:
        Z, Ul = util.apply_PCA_from_Eig(X, U, l, mu)
        X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
        util.plot_gallery(X_rec[:12])

    ### ========== TODO: END ========== ###


    #===============================================
    # (Optional) part 2: test Cluster implementation
    # centroid: [ 1.04022358  0.62914619]
    # medoid:   [ 1.05674064  0.71183522]

    np.random.seed(1234)
    sim_points = generate_points_2d(20)
    cluster = Cluster(sim_points)
    print('centroid:', cluster.centroid().attrs)
    print('medoid:', cluster.medoid().attrs)

    # part 2: test kMeans and kMedoids implementation using toy dataset
    np.random.seed(1234)



    ### ========== TODO: START ========== ###
    # part 3a: cluster faces
    np.random.seed(1234)
    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)
    k_means = []
    k_medoids = []

    for i in range(10):
        kmeans = kMeans(points, 4, init="random")
        k_means.append(kmeans.score())

    Mean = np.mean(k_means)
    Minimum = min(k_means)
    Maximum = max(k_means)

    print("Mean, Minimum and Maximum for Kmeans: ", Mean, Minimum, Maximum)

    for i in range(10):
        kmedoids = kMedoids(points, 4, init="random")
        k_medoids.append(kmedoids.score())

    Mean = np.mean(k_medoids)
    Minimum = min(k_medoids)
    Maximum = max(k_medoids)

    print("Mean, Min and Max for kmedoids are : ", Mean, Minimum, Maximum)



    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)
    U, mu = util.PCA(X)
    X1, y1 = util.limit_pics(X, y, [4, 13], 40)
    l = np.arange(1, 42, 2)
    k_means = []
    k_medoids = []
    for i in l:
        Z, Ul = util.apply_PCA_from_Eig(X1, U, i, mu)
        X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
        points = build_face_image_points(X_rec, y1)
        kmeans = kMeans(points, 2, init='cheat')
        k_means.append(kmeans.score())
        kmedoids = kMedoids(points, 2, init='cheat')
        k_medoids.append(kmedoids.score())


    plt.plot(l, k_means, label='K-means')
    plt.plot(l, k_medoids, label='K-medoids')
    plt.title('K-means and K-medoids score vs, number of principal components')
    plt.xlabel('Number of principal components')
    plt.ylabel('Clustering score')
    plt.legend()
    plt.show()


    # part 3c: determine "most discriminative" and "least discriminative" pairs of images
    np.random.seed(1234)
    best_score = -1
    worst_score = np.inf
    best_score_pair = None, None
    worst_score_pair = None, None
    for i in range(19):
        for j in range(19):
            if i != j:
                X_ij, y_ij = util.limit_pics(X, y, [i,j], 40)
                points = build_face_image_points(X_ij, y_ij)
                score = kMedoids(points, 2, init='cheat').score() # KMedoids better than KMeans
                if score < worst_score:                             # cheat_init --> fewer iters
                    worst_score = score
                    worst_score_pair = i, j
                if score > best_score:
                    best_score = score
                    best_score_pair = i, j

    print("Best pair with the score : ", best_score_pair, best_score)
    s = list(best_score_pair)
    X1, y1 = util.limit_pics(X, y, s, 40)
    for i in range(2):
        util.show_image(X1[i]
)

    print("Worst pair with the score : ", worst_score_pair, worst_score)
    s = list(worst_score_pair)
    X2, y2 = util.limit_pics(X, y, s, 40)
    for i in range(2):
        util.show_image(X2[i])
    ### ========== TODO: END ========== ###


if __name__ == "__main__":
    main()
