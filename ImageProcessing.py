import cv2
import numpy as np

import Clustering


def run_surf(img_path, value=400):
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img2 = cv2.fastNlMeansDenoising(img, None, 10)
    des = None
    kp = None

    attempts = 0
    while des is None and attempts < 10:
        surf = cv2.SURF(value)
        kp, des = surf.detectAndCompute(img2, None)

        if kp is not None and des is not None:
            kp, des = filter_kps(kp, des, img2)
            break

        value *= 0.9
        print "SURF detected no features for %s re-running at %f" % (img_path, value)
        attempts += 1

    return kp, des


def filter_kps(key_points, descriptors, img):
    filtered_kps = []
    indices_to_delete = []
    for index, kp in enumerate(key_points):
        radius = kp.size / 2.0
        key_point_area = img[kp.pt[1] - radius: kp.pt[1] + radius, kp.pt[0] - radius:kp.pt[0] + radius]
        white_pixels = key_point_area[np.where(key_point_area > 150)]

        if white_pixels.size / float(key_point_area.size) < 0.1:
            filtered_kps.append(kp)
        else:
            indices_to_delete.append(index)

    return filtered_kps, np.delete(descriptors, indices_to_delete)


def get_coordinates(key_points):
    return map(lambda x: x.pt, key_points)


def crop_image(img_path, output_path, key_points, min_k):
    print "Cropping image %s" % img_path

    coordinates = get_coordinates(key_points)
    db = Clustering.run_dbscan(coordinates, min_k)
    y1, y2, x1, x2 = build_bounding_box(db, coordinates, min_k)

    img = cv2.imread(img_path)
    cropped_image = img[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)


def build_bounding_box(db, coordinates, min_k):
    x_coordinates = []
    y_coordinates = []

    clusters, histograms = Clustering.build_cluster_histograms(db)

    cluster_index = 0
    while len(x_coordinates) < len(coordinates) * 0.4:
        for index, label in enumerate(db.labels_):
            if label == clusters[cluster_index]:
                x_coordinates.append(coordinates[index][0])
                y_coordinates.append(coordinates[index][1])
        cluster_index += 1
        if cluster_index >= len(clusters):
            if len(x_coordinates) < min_k * 2:
                x_coordinates = map(lambda x: x[0], coordinates)
                y_coordinates = map(lambda x: x[1], coordinates)
            break

    return min(y_coordinates), max(y_coordinates), min(x_coordinates), max(x_coordinates)
