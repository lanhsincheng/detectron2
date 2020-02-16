import os
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from fvcore.common.file_io import PathManager
import xml.etree.ElementTree as ET
import csv

# __all__ = ["get_lung_dicts","register_lung_dataset", "register_all_lung_dataset"]

# fmt: off
CLASS_NAMES = [
    "nodule",
]
# fmt: on

# dirname include Annotations and JPEGImages files
dirname = r'D:\lung_data_laptop\_lung_data_laptop\1203_data\1cls_res50_gray_train'
test_dirname = r'D:\lung_data_laptop\_lung_data_laptop\1203_data\1cls_res50_gray_test'
# name to register for newly added datasets
name = "lung_dataset"

# csv root, for reading train and val parts purpose
root_of_data_csv_path = r'./training_data_csv/lung_dataset/'
# train_data_csv_path = r'./training_data_csv/lung_dataset/train.csv'
# val_data_csv_path = r'./training_data_csv/lung_dataset/val.csv'
# trainval_data_csv_path = r'./training_data_csv/lung_dataset/trainval.csv'
# test_data_csv_path = r'./training_data_csv/lung_dataset/test.csv'

def get_train_dataset_path(data_csv_path):

    fileids = []
    with open(data_csv_path, newline='') as csvFile:
        rows = csv.reader(csvFile)
        for row in rows:
            # print(os.path.split(row[0])[1].split('.')[0])
            fileids.append(os.path.split(row[0])[1].split('.')[0])
    return fileids

def get_lung_dicts(dirname,data_csv_path):

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

def register_lung_dataset(name, dirname, data_csv_path):

    DatasetCatalog.register(name, lambda: get_lung_dicts(dirname, data_csv_path))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname
    )

# ==== Predefined splits for lung dataset ===========
def register_all_lung_dataset(root_of_data_csv_path, dirname, test_dirname): #root_of_data_csv_path = r'./training_data_csv/lung_dataset/'
    SPLITS = [
        ("lung_dataset_train", dirname, "train.csv"),
        ("lung_dataset_val", dirname, "val.csv"),
        ("lung_dataset_trainval", dirname, "trainval.csv"),
        ("lung_dataset_test", test_dirname, "test.csv"),
    ]
    for name, dirname, split in SPLITS:
        # print(name, dirname, root_of_data_csv_path + '/' + split)
        register_lung_dataset(name, dirname, root_of_data_csv_path + '/' + split)
        if name == "lung_dataset_trainval":
            lung_metadata = MetadataCatalog.get("lung_dataset_trainval")
        # MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    return lung_metadata

# register_all_lung_dataset(root_of_data_csv_path, dirname, test_dirname)
# get_train_dataset_path(test_data_csv_path)