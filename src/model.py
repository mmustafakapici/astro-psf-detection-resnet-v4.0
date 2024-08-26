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

class myadvanced_fasterRCNN_model(torch.nn.Module):
    def __init__(self, num_classes):
        super(myadvanced_fasterRCNN_model, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        return self.model(images, targets)
    
    def predict(self, images):
        self.eval()
        return self.model(images)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
    
    def freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def freeze_rpn(self):
        for param in self.model.rpn.parameters():
            param.requires_grad = False

    def unfreeze_rpn(self):
        for param in self.model.rpn.parameters():
            param.requires_grad = True

    def freeze_roi_heads(self):
        for param in self.model.roi_heads.parameters():
            param.requires_grad = False

    def unfreeze_roi_heads(self):
        for param in self.model.roi_heads.parameters():
            param.requires_grad = True

    def freeze_all(self):
        self.freeze_backbone()
        self.freeze_rpn()
        self.freeze_roi_heads()

    def unfreeze_all(self):
        self.unfreeze_backbone()
        self.unfreeze_rpn()
        self.unfreeze_roi_heads()

    def get_backbone(self):
        return self.model.backbone

    def get_rpn(self):
        return self.model.rpn

    def get_roi_heads(self):
        return self.model.roi_heads