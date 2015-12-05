import cv2
import os
import numpy as np

import Clustering


def run_dense_surf(img_path, value=400):
    if type(img_path) == str and os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = img_path

    dense = cv2.FeatureDetector_create("Dense")
    des = None
    kp = None
    attempts = 0
    while des is None and attempts < 10:
        surf = cv2.SURF(value)
        kp = dense.detect(img)
        kp, des = surf.compute(img, kp)

        if kp is not None and des is not None:
            break

        value *= 0.9
        print "SURF detected no features for %s re-running at %f" % (img_path, value)
        attempts += 1

    return kp, des


def run_surf(img_path, value=400):
    if type(img_path) == str and os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = img_path

    des = None
    kp = None
    attempts = 0
    while des is None and attempts < 10:
        surf = cv2.SURF(value)
        kp, des = surf.detectAndCompute(img, None)

        if kp is not None and des is not None:
            break

        value *= 0.9
        print "SURF detected no features for %s re-running at %f" % (img_path, value)
        attempts += 1

    return kp, des


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


def get_color_histogram(image):
    return cv2.calcHist( [image], [0, 1], None, [180, 256], [0, 180, 0, 256] )


def divide_image(image):
    height, width, channels = image.shape
    section1 = image[0:height / 2, 0:width / 2]
    section2 = image[0:height / 2, width / 2:width]
    section3 = image[height / 2:height, 0:width / 2]
    section4 = image[height / 2:height, width / 2:width]
    return [section1, section2, section3, section4]


def crop_by_color_histogram(image, output_path):
    """
    From: https://github.com/eduardofv/whale_detector/blob/master/hist_zones.py
    """
    print "Cropping image %s" % image
    im = cv2.imread(image)
    original_image = np.copy(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hist = get_color_histogram(im)
    mask_image(im, hist)

    cleaned_image = clean_image(im)
    best_region = find_best_region(cleaned_image)
    cropped_image = crop_image_from_region(original_image, best_region)

    cv2.imwrite(output_path, cropped_image)


def set_color(image, color):
    for i in range(0, 3):
        image[:, :, i] = color


def mask_image(image, hist):
    height, width, channels = image.shape
    if width < 10 or height < 10:
        set_color(image, 0)
        return

    sectioned_image = divide_image(image)
    correlation = [cv2.compareHist(hist, get_color_histogram(imx), cv2.cv.CV_COMP_CORREL) for imx in sectioned_image]

    for index, section in enumerate(sectioned_image):
        if correlation[index] < 0.25:
            set_color(section, 255)
        else:
            correlation.append(mask_image(section, hist))

    return correlation


def clean_image(image):
    kernel = np.ones((40, 40), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return cv2.dilate(image, kernel, iterations=1)


def find_best_region(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_size = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        size = w * h
        if size > best_size:
            best_size = size
            best = cnt

    return best


def crop_image_from_region(image, region):
    x, y, w, h = cv2.boundingRect(region)
    return image[y: y + h, x: x + w]
