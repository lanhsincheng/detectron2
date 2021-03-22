# from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import copy
import torch
# from detectron2.data import get_detection_dataset_dicts

def resize_mammo_mapper(dataset_dict):
    # print('use resize_mammo_mapper')
    dataset_dict = copy.deepcopy(dataset_dict)  # this will be modified by following code
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    image, transforms = T.apply_transform_gens([T.RandomFlip(), T.ResizeShortestEdge([640, 672, 704, 736, 768, 800], 1333, sample_style="choice")], image) # T.RandomContrast(1, 2), T.RandomBrightness(0.5, 1.5), T.RandomRotation([90, 180, 270], sample_style ="choice")
    # print("transformation applied: ", transforms)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict



