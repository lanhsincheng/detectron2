# from lung_dataset import get_lung_dicts, register_all_lung_dataset, register_lung_dataset
from projects.lung_project.lung_dataset import *
import cv2
import random
from detectron2.utils.visualizer import Visualizer

# ==== verify the data loading is correct ===========

# dirname include Annotations and JPEGImages files
dirname = r'D:\lung_data_laptop\_lung_data_laptop\1203_data\1cls_res50_gray_train'
test_dirname = r'D:\lung_data_laptop\_lung_data_laptop\1203_data\1cls_res50_gray_test'
# csv root, for reading train and val parts purpose
root_of_data_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\lung_dataset/'
train_data_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\lung_dataset/train.csv'
val_data_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\lung_dataset/test.csv'


dataset_dicts = get_lung_dicts(dirname,train_data_csv_path)
lung_metadata = register_all_lung_dataset(root_of_data_csv_path, dirname, test_dirname)
for d in random.sample(dataset_dicts, 20):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=lung_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('My result',vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()