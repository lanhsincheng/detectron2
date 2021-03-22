from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import xlsxwriter
from projects.mammography_project.mammo_dataset import *
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator, PascalVOCDetectionEvaluator, MammoDatasetEvaluator
from detectron2.engine import DefaultPredictor, DefaultExport
from detectron2.data import build_detection_test_loader
from demo.mammo_visualize_demo import mammo_visualize_demo

from projects.mammography_project.mammo_dataloader import mammo_mapper
from detectron2.data import build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
from PIL import Image


setup_logger()
test_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset\test.csv'
# train_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\mammo_dataset/train.csv'
# test_csv_path = r'D:\PycharmProjects\detectron2\training_data_csv\INbreast\test.csv'
mammo_metadata = register_all_mammo_dataset(root_of_data_csv_path, dirname, dirname)
cfg = get_cfg()
# dataset_dict_list = get_mammo_dicts(dirname,train_csv_path)

def visualize_dataloader(cfg,trainer):
    def output(vis, fname):
        filepath = os.path.join(r'D:\PycharmProjects\detectron2\projects\mammography_project\visualize_dataloader/', fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)
    print("start visualize....")
    train_data_loader = build_detection_train_loader(cfg, mapper=mammo_mapper)
    for batch in train_data_loader:
        for per_image in batch:
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0)
            if cfg.INPUT.FORMAT == "BGR":
                img = img[:, :, [2, 1, 0]]
            else:
                img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

            visualizer = Visualizer(img, metadata=mammo_metadata, scale=1.0)
            target_fields = per_image["instances"].get_fields()
            labels = [mammo_metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            output(vis, str(per_image["image_id"]) + ".jpg")


# ==========================================================================================================
#mammo logs
#pretrain model default path "detectron2.model_zoo.configs
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

#==== set parameter in default.py ===================
cfg.DATASETS.TRAIN = ("mammo_dataset_train", )
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0 #WINDOWS REMEMBER TO SET THIS 0

# Detectron2 recognizes models in pytorch’s .pth format, as well as the .pkl files in our model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo, if is empty means initializing from scratch
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = r"D:\PycharmProjects\detectron2\tools\converted_trim_model/modified.pth"

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025   # pick a good LR
cfg.SOLVER.STEPS = (185000, 195000) # (35000, 45000)
cfg.SOLVER.MAX_ITER = 200000    # 50000 iterations seems good enough for this dataset; you may need to train longer for a practical dataset
iteration = 200000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # only has one class (nodule)
# =============================================

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# ==== train part ===================
trainer = DefaultTrainer(cfg)
visualize_dataloader(cfg,trainer)
trainer.resume_or_load(resume=False)
trainer.train()

# ==== testset evaluation for final checkpoint part ===========
print('==========testset evaluation for final checkpoint part==========')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.WEIGHTS = "D:\PycharmProjects\detectron2\output\model final HISTORY\mammo_inbreast0306_PRETRAIN_faster_rcnn_R_50_fpn_model_final/model_0079999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
cfg.DATASETS.TEST = ("mammo_dataset_test",)
predictor = DefaultPredictor(cfg)

# visualize prediction on test image
mammo_visualize_demo(dirname, predictor, mammo_metadata, test_csv_path, output_dir=r'D:\Mammograph\result\faster_rcnn_R50_inbreast0309_visualize_0321')

evaluator = MammoDatasetEvaluator("mammo_dataset_test")
val_loader = build_detection_test_loader(cfg, "mammo_dataset_test")
# meth'inference_on_dataset'. The return value of `evaluator.evaluate()
inference_on_dataset(predictor.model, val_loader, evaluator)
# another equivalent way is to use trainer.test

# ==== testset evaluation for every checkpoints(except final one) part ===========
print('==========testset evaluation for every checkpoints(except final one) part==========')
pth_file = os.listdir(cfg.OUTPUT_DIR)
checkpoint_list = []
mAP50_for_checckpoint_list = []
for file in pth_file:
    if file.split('_')[0] == 'model':
        checkpoint_list.append(file)
print(checkpoint_list)
range_value = int(iteration/5000)+1 # the val to put in following range, or you can set it manually(check how many pth in dir)
for iter_checkpoint in range(20):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, checkpoint_list[iter_checkpoint])
    print(os.path.join(cfg.OUTPUT_DIR, checkpoint_list[iter_checkpoint]))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("mammo_dataset_test", )
    predictor = DefaultPredictor(cfg)

    evaluator = MammoDatasetEvaluator("mammo_dataset_test")
    val_loader = build_detection_test_loader(cfg, "mammo_dataset_test")
    # meth'inference_on_dataset'. The return value of `evaluator.evaluate()
    ret, mAP_50_RECORD = inference_on_dataset(predictor.model, val_loader, evaluator)
    mAP50_for_checckpoint_list.append(mAP_50_RECORD)
# write AP50 to the xlsfile
workbook = xlsxwriter.Workbook('0321every _checkpoint_mAP50.xlsx')
worksheet = workbook.add_worksheet()
row = 0
column = 0
for item in mAP50_for_checckpoint_list:
    # write operation perform
    worksheet.write(row, column, item)
    column += 1
workbook.close()