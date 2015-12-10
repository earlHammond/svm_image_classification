import numpy as np
import gc

from sklearn.cluster import MiniBatchKMeans, KMeans


def get_kmeans_cluster(k, cluster_centers):
    return KMeans(n_clusters=k, max_iter=100, init=cluster_centers, n_init=1)


def get_mini_batch_cluster(k):
    return MiniBatchKMeans(init='k-means++', n_clusters=k, max_iter=10, batch_size=10)


def cluster_key_points_from_stream(image_data, k):
    print "Clustering Image Features to %s clusters" % k
    km = get_mini_batch_cluster(k)

    descriptors = []
    for index, (kp, des) in enumerate(image_data):
        descriptors.append(des)

        if index % 10 == 0:
            descriptors = np.vstack(descriptors)
            km.partial_fit(descriptors)
            descriptors = []
            gc.collect()

    if descriptors:
        descriptors = np.vstack(descriptors)
        km.partial_fit(descriptors)

    return km.cluster_centers_
