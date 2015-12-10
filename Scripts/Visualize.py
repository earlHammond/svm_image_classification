import cv2
from matplotlib import pyplot as plt

import ImageProcessing


def save_image_with_key_points(img, kp, output_path):
    image_with_keys = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
    plt.interactive(False)
    plt.imshow(image_with_keys)
    plt.savefig(output_path)


def draw_plot(points):
    plt.scatter(points)
    plt.show()


# image_path = "/Users/computer/School/UChicago/Data_Mining_2/Project/Data/Input/imgs_test/w_4.jpg"
# kp, des = ImageProcessing.run_surf(image_path)
#
# print len(des)
# img = cv2.imread(image_path)
# save_image_with_key_points(img, kp, "/Users/computer/Desktop/test_image.jpg")

img_path = "/Users/computer/Desktop/w_676.jpg"
out_path = "/Users/computer/Desktop/w_676_c.jpg"
img = cv2.imread(img_path)

kps, des = ImageProcessing.run_surf(img_path)


image_with_keys = cv2.drawKeypoints(img, kps, None, (255, 0, 0), 4)
plt.imshow(image_with_keys)
plt.show()
ImageProcessing.crop_image(img_path, out_path, kps, 50)


