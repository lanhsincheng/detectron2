import os
import random
import csv

"""
To create csv imageset for INbreast dataset(without duplicate) for VOC format code to transform as inoter custom dataset
"""
xml_file_path = r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\AllXML/'
jpg_root = r'D:\Mammograph\reference_dataset\INbreast Release 1.0\INbreast Release 1.0\JPEGImages/'
xml_dir = os.listdir(xml_file_path)
f = open('train_imageset.csv', 'w', newline='')
with f:
    writer = csv.writer(f)

    for row in xml_dir:
        context = jpg_root + row.split('.')[0] + '.jpg'
        writer.writerow([context])

