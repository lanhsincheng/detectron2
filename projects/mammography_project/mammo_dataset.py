import os
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from fvcore.common.file_io import PathManager
import xml.etree.ElementTree as ET
import csv

# fmt: off
# CLASS_NAMES = [
#     "Asymmetry", "Calcification", "Cluster", "Spiculated Region", "Distortion", "Mass", "Point 3", "Point 1", "Unnamed",
# ]
# CLASS_NAMES = [
#     "malignant",
# ]
CLASS_NAMES = [
    "benign", "malignant",
]
# CLASS_NAMES = [
#     "benign", "malignant", "benign0gabor", "malignant0gabor", "benign1gabor", "malignant1gabor", "benign2gabor", "malignant2gabor",
# ]
# CLASS_NAMES = [
#     "0", "1",
# ]
# fmt: on

# dirname include Annotations and JPEGImages files
# dirname = r'D:\Mammograph\training_dataset'
dirname = r'D:\Mammograph\ROI_training_dataset'
# dirname = r'D:\Mammograph\gabor_training_dataset'
# dirname = r'D:\Mammograph\DDSM_training_dataset'
# dirname = r'D:\Mammograph\concate_training_dataset'
# dirname = r'D:\Mammograph\reference_dataset\INbreast Release 1.0'

# name to register for newly added datasets
name = "mammo_dataset"
# csv root, for reading train and val parts purpose
root_of_data_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset/'
# root_of_data_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\INbreast/'

def get_train_dataset_path(data_csv_path):

    fileids = []
    with open(data_csv_path, newline='') as csvFile:
        rows = csv.reader(csvFile)
        for row in rows:
            # print(os.path.split(row[0])[1].split('.')[0])
            fileids.append(os.path.split(row[0])[1].split('.')[0])
    return fileids

def get_mammo_dicts(dirname,data_csv_path):

    fileids = get_train_dataset_path(data_csv_path)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml") #D:\pascal voc\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations\000005.xml
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg") #D:\pascal voc\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000005.jpg

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }

        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls==None:
                print(fileid)
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        #r{dict}{
                # "file_name": 'D:\pascal voc\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000012.jpg',
                # "image_id": 000012,
                # "height": 333,
                # "width": 500,
                # "annptations": [{'category_id': 6, 'bbox': [155.0, 96.0, 351.0, 270.0], 'bbox_mode': <BoxMode.XYXY_ABS: 0>},{'category_id': 6, 'bbox': [15.0, 6.0, 51.0, 70.0], 'bbox_mode': <BoxMode.XYXY_ABS: 0>}]
        #       }
        dicts.append(r)
    return dicts

def register_mammo_dataset(name, dirname, data_csv_path):

    DatasetCatalog.register(name, lambda: get_mammo_dicts(dirname, data_csv_path))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname
    )

# ==== Predefined splits for mammo dataset ===========
def register_all_mammo_dataset(root_of_data_csv_path, dirname, test_dirname): #root_of_data_csv_path = r'./training_data_csv/mammo_dataset/'
    SPLITS = [
        ("mammo_dataset_train", dirname, "train.csv"),
        ("mammo_dataset_test", dirname, "test.csv"),
    ]
    for name, dirname, split in SPLITS:
        # print(name, dirname, root_of_data_csv_path + '/' + split)
        register_mammo_dataset(name, dirname, root_of_data_csv_path + '/' + split) #dirname = r'D:\Mammograph\training_dataset'
        if name == "mammo_dataset_train":
            mammo_metadata = MetadataCatalog.get("mammo_dataset_train")
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    return mammo_metadata

# register_all_mammo_dataset(root_of_data_csv_path, dirname, dirname)
# get_train_dataset_path(test_data_csv_path)