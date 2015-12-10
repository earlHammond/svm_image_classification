import os
import sys
import multiprocessing as mp

from src import FileUtils
from src import ImageProcessing


def extract_image_features_stream_parallel(image):
    print "Extracting SURF for image %s" % image
    kp, des = ImageProcessing.run_surf(image, spread="sparse", k=100)


pool = mp.Pool(4)
output_path = sys.argv[1]

sorted_images = os.path.join(output_path, "SortedImages")
images = FileUtils.get_images(sorted_images)
print images

pool.map(extract_image_features_stream_parallel, images)
pool.close()
pool.join()
