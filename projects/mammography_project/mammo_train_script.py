from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from projects.mammography_project.mammo_dataset import *
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator, PascalVOCDetectionEvaluator, MammoDatasetEvaluator
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from demo.mammo_visualize_demo import mammo_visualize_demo
setup_logger()

# dirname include Annotations and JPEGImages files
# dirname = r'D:\lung_data_laptop\_lung_data_laptop\1203_data\1cls_res50_gray_train'
# test_dirname = r'D:\lung_data_laptop\_lung_data_laptop\1203_data\1cls_res50_gray_test'
# csv root, for reading train and val parts purpose
# root_of_data_csv_path = r'./training_data_csv/lung_dataset/'
# test_data_csv_path = r'./training_data_csv/lung_dataset/test.csv'
test_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset\test.csv'

mammo_metadata = register_all_mammo_dataset(root_of_data_csv_path, dirname, dirname)

cfg = get_cfg()
#lung logs
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
#mammo logs
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("mammo_dataset_train", )
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo, if is empty means initializing from scratch
# cfg.MODEL.WEIGHTS = "D:\PycharmProjects\detectron2\projects\mammography_project\output/faster_rcnn_R_101_fpn_model_final.pth"
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
cfg.DATASETS.TEST = ("mammo_dataset_test", )
predictor = DefaultPredictor(cfg)

# visualize prediction on test image
mammo_visualize_demo(dirname, predictor, mammo_metadata, test_csv_path, output_dir=r'D:\Mammograph\result\faster_rcnn_R50_NO_PRETRAIN_visualize_0219')

evaluator = MammoDatasetEvaluator("mammo_dataset_test")
val_loader = build_detection_test_loader(cfg, "mammo_dataset_test")
# meth'inference_on_dataset'. The return value of `evaluator.evaluate()
inference_on_dataset(predictor.model, val_loader, evaluator)
# another equivalent way is to use trainer.test

