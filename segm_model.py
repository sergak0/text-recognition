import cv2
import torch
import numpy as np
import mask_creation
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_contours_from_mask(mask, min_area=100):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE
                                           )
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


class SEGMpredictor:
    def __init__(self, model_path):
         # self.model = get_instance_segmentation_model(2).to(device)
         self.model = torch.load(model_path, map_location=device)
         self.model = self.model.eval()
    
    def __call__(self, image, return_mask=False):
        image = image.copy()
        mask = mask_creation.predict_full_img(image, self.model, device)
        mask_post = mask != 0
        
        # mask_post = mask_postprocessing(image, mask_post)
        ans = get_contours_from_mask(mask_post)

        if return_mask:
          return mask, ans
        
        return ans

