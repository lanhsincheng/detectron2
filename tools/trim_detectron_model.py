import os
import torch
import argparse
from detectron2.config import get_cfg

"""
To trim last layers' parameter, becasuse as pretrain model, different class numbers' weight might affect result of target dataset training
e.g. one class weight dominate finune model class' weights
"""

cfg = get_cfg()
modified_model_path = r"C:\Users\Lan-Hsin Cheng\.cache\torch\checkpoints/tf_efficientnet_b6_aa-80ba17e4.pth"
save_model_path = r"./converted_trimC:\Users\Lan-Hsin Cheng\.cache\torch\checkpoints/tf_efficientnet_b6_aa-80ba17e4.pth"
corresponding_yaml_path = r"D:\PycharmProjects\pytorch-image-models\output\train\20200710-163606-tf_efficientnet_b6-528/args.yaml"

def removekey(d, listofkeys): # listofkeys = ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight']
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    # default="~/.torch/models/_detectron_35858933_12_2017_baselines_e2e_mask_rcnn_R-50-FPN_1x.yaml.01_48_14.DzEQe4wC_output_train_coco_2014_train%3Acoco_2014_valminusminival_generalized_rcnn_model_final.pkl",
    default=modified_model_path,
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    # default="./pretrained_model/mask_rcnn_R-50-FPN_1x_detectron_no_last_layers.pth",
    default=save_model_path,
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    # default="configs/e2e_mask_rcnn_R_50_FPN_1x.yaml",
    default=corresponding_yaml_path,
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(args.cfg)
_d = torch.load(DETECTRON_PATH)
print(_d)
# _d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

newdict['model'] = removekey(_d['model'], ['roi_heads.box_predictor.cls_score.bias', 'roi_heads.box_predictor.cls_score.weight', 'roi_heads.box_predictor.bbox_pred.bias', 'roi_heads.box_predictor.bbox_pred.weight'])
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))