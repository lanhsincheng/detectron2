# from lung_dataset import get_lung_dicts, register_all_lung_dataset, register_lung_dataset
from projects.mammography_project.mammo_dataset import *
import cv2
import random
from detectron2.utils.visualizer import Visualizer

# ==== verify the data loading is correct ===========

train_data_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset/train.csv'
# train_data_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\INbreast/train.csv'
dataset_dicts = get_mammo_dicts(dirname,train_data_csv_path)
mammo_metadata = register_all_mammo_dataset(root_of_data_csv_path, dirname, dirname)
for d in random.sample(dataset_dicts, 13):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=mammo_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    # cv2.imwrite(r'D:\PycharmProjects\detectron2\training_data_csv\INbreast\debug_img/' + 'debug.jpg', vis.get_image()[:, :, ::-1])
    cv2.imshow(d["file_name"],vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()