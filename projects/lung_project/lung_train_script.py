from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from projects.lung_project.lung_dataset import *
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator, PascalVOCDetectionEvaluator, LungDatasetEvaluator
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from demo.lung_visualize_demo import lung_visualize_demo
setup_logger()

# dirname include Annotations and JPEGImages files
dirname = r'D:\lung_data_laptop\_lung_data_laptop\1203_data\1cls_res50_gray_train'
test_dirname = r'D:\lung_data_laptop\_lung_data_laptop\1203_data\1cls_res50_gray_test'
# csv root, for reading train and val parts purpose
root_of_data_csv_path = r'./training_data_csv/lung_dataset/'
test_data_csv_path = r'./training_data_csv/lung_dataset/test.csv'

lung_metadata = register_all_lung_dataset(root_of_data_csv_path, dirname, test_dirname)

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("lung_dataset_trainval", )
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025   # pick a good LR
cfg.SOLVER.STEPS = (35000, 45000)
cfg.SOLVER.MAX_ITER = 50000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (nodule)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# ==== train part ===================
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ==== testset evaluation part ===========
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("lung_dataset_test", )
predictor = DefaultPredictor(cfg)

# visualize prediction on test image
lung_visualize_demo(test_dirname, predictor, lung_metadata, test_data_csv_path, output_dir=r'D:\lung_data_laptop\_lung_data_laptop\1203_data\retinanet_visualize_R_101_visualize_0213')

evaluator = LungDatasetEvaluator("lung_dataset_test")
val_loader = build_detection_test_loader(cfg, "lung_dataset_test")
# meth'inference_on_dataset'. The return value of `evaluator.evaluate()
inference_on_dataset(predictor.model, val_loader, evaluator)
# another equivalent way is to use trainer.test

