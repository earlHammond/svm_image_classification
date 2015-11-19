import cv2

import Clustering


def run_surf(img_path):
    img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(img, None)
    return kp, des


def get_coordinates(key_points):
    return map(lambda x: x.pt, key_points)


def crop_image(img_path, output_path, key_points, min_k):
    print "Cropping image %s" % img_path

    coordinates = get_coordinates(key_points)
    db = Clustering.run_dbscan(coordinates)
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


