# from detectron2.utils.visualizer import ColorMode
import cv2
import random
from detectron2.utils.visualizer import Visualizer
from projects.lung_project.lung_dataset import *


def lung_visualize_demo(test_dirname, predictor, dataset_metadata, test_data_csv_path, output_dir):
    # for d in random.sample(dataset_dicts, 3):
    dataset_dicts = get_lung_dicts(test_dirname,test_data_csv_path)
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=0.8,
                       # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(output_dir + '/' + d["image_id"] + '.jpg', v.get_image()[:, :, ::-1])
        # cv2.imshow('My result', v.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()