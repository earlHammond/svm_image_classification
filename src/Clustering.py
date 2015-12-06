import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans


def get_kmeans_cluster(k, cluster_centers):
    return KMeans(n_clusters=k, max_iter=100, init=cluster_centers, n_init=1)


def get_mini_batch_cluster(k):
    return MiniBatchKMeans(init='k-means++', n_clusters=k, max_iter=100, batch_size=50)


def cluster_key_points_from_stream(image_data, k):
    print "Clustering Image Features to %s clusters" % k
    km = get_mini_batch_cluster(k)

    descriptors = []
    for index, data in enumerate(image_data):
        descriptors.append(data[1])

        if index % 100:
            descriptors = np.vstack(descriptors)
            km.partial_fit(descriptors)
            descriptors = []

    return km.cluster_centers_
