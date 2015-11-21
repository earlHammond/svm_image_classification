import os
import shutil


image_path = "/Users/computer/School/UChicago/Data_Mining_2/Project/TestData/Training1"
input_path = "/Users/computer/School/UChicago/Data_Mining_2/Project/DataAll/Input"
output_path = "/Users/computer/School/UChicago/Data_Mining_2/Project/TestData/Input/imgs_test"

for root, dirs, files in os.walk(image_path):
    for f in files:
        if 'jpg' in os.path.splitext(f)[1]:
            source_path = os.path.join(input_path, f)
            target_path = os.path.join(output_path, f)
            shutil.copy(source_path, target_path)
