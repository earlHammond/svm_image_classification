import os

import cv2
import numpy as np


def load_image(image_path):
    if type(image_path) == str and os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)
        # img = cv2.fastNlMeansDenoisingColored(img)
    elif type(image_path) == np.ndarray:
        img = image_path
    else:
        raise Exception("Invalid image found.")

    return img


def detect_compute_surf(img, surf, spread):
    if spread == 'dense':
        dense = cv2.FeatureDetector_create("Dense")
        kp = dense.detect(img)
        kp, des = surf.compute(img, kp)
    else:
        kp, des = surf.detectAndCompute(img, None)

    return kp, des


def run_surf(image_path, spread='dense', value=400):
    img = load_image(image_path)

    des = None
    kp = None
    attempts = 0
    while des is None:
        surf = cv2.SURF(value)
        kp, des = detect_compute_surf(img, surf, spread)

        if kp is not None and des is not None:
            break

        value *= 0.9
        print "SURF detected no features for %s re-running at %f" % (image_path, value)
        attempts += 1

    return kp, des


def crop_by_color_histogram(image, output_path):
    """
    From: https://github.com/eduardofv/whale_detector/blob/master/hist_zones.py
    """
    print "Cropping image %s" % image
    img = load_image(image)
    original_image = np.copy(img)

    im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = get_color_histogram(im)
    mask_image(im, hist)

    cleaned_image = clean_image(im)
    best_region = find_best_region(cleaned_image)
    cropped_image = crop_image_from_region(original_image, best_region)

    cv2.imwrite(output_path, cropped_image)


def get_color_histogram(image):
    return cv2.calcHist( [image], [0, 1], None, [180, 256], [0, 180, 0, 256] )


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


def set_color(image, color):
    for i in range(0, 3):
        image[:, :, i] = color


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


def divide_image(image):
    try:
        height, width, _ = image.shape
    except ValueError:
        height, width = image.shape

    section1 = image[0:height / 2, 0:width / 2]
    section2 = image[0:height / 2, width / 2:width]
    section3 = image[height / 2:height, 0:width / 2]
    section4 = image[height / 2:height, width / 2:width]
    return [section1, section2, section3, section4]


def crop_image_from_region(image, region):
    x, y, w, h = cv2.boundingRect(region)
    return image[y: y + h, x: x + w]

