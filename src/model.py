import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor ,FasterRCNN_ResNet50_FPN_Weights
from utils import load_config

config = load_config()


weights  =  config["model"]["pretrained_weights"]


if weights == "COCO":
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
#default
else:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

def faster_rcnn_resnet50_model(num_classes):
    # COCO dataset için önceden eğitilmiş Faster R-CNN modelini yükleyin
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Son katmanı (başlığı) değiştirelim
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
