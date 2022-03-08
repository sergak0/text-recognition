import json
import os
import sys
import warnings

import cv2
import numpy as np
import logging
import torch
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image
from segm_model import SEGMpredictor
from trocr_model import TrOcrModel
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import re


TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SEGM_MODEL_PATH = "mask_rcnn_small_adam_en_best"
OCR_EN_MODEL_PATH = "tr_ocr_best_eng"
OCR_RU_MODEL_PATH = "tr_ocr_best_small"
OCR_MULTI_MODEL_PATH = "tr_ocr_best_multilingual"
CLASSIFIER_MODEL_PATH = "language_classifier"

def crop_img_by_polygon(img, polygon):
    # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst

def get_polygon_for_answer(polygon, croped):
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x1,y1,w,h = rect
    mid_x = x1 + w // 2
    mid_y = y1 + h // 2

    best = 1e9
    for i in range(h):
      now = abs((croped[:i, :] != [0, 0, 0]).sum() - (croped[i:, :] != [0, 0, 0]).sum())
      if now < best:
        best = now
        mid_y = i + y1

    x1 = mid_x - w // 5
    x2 = mid_x + w // 5
    # return [(mid_y, x1), (mid_y + 5, x1), (mid_y + 5, x2), (mid_y, x2)]
    return [(x1, mid_y), (x1, mid_y + 1), (x2, mid_y + 1), (x2, mid_y)]

def get_classifier_model():

  model = models.resnet50(pretrained=True)
  model.fc = nn.Sequential(
      nn.Linear(2048, 256),
      nn.ReLU(),
      nn.Linear(256, 2)
  )
  return model
  
  
class PiepleinePredictor:
    def __init__(self, segm_model_path, ru_ocr_model_path, en_ocr_model_path, classifier_model_path, multilingual_model_path):
        self.seg_model = SEGMpredictor(segm_model_path)
        self.text_model = {'ru': TrOcrModel(ru_ocr_model_path),
                           'en': TrOcrModel(en_ocr_model_path, False),
                           'multiling': TrOcrModel(multilingual_model_path, False)}
        self.batch_size = 50
        self.transforms = A.RandomScale(scale_limit=(-0.5, -0.5), p=1)

        self.classifier_model = torch.load(classifier_model_path, map_location=device)
        self.classifier_predicts = 5
        self.classifier_transforms = A.Compose([
                  A.Resize(128, 384),
                  ToTensor()
              ])

    def predict_language(self, images):
        p = np.argsort([el.shape[1] for el in images])
        batch = []
        for idx in p[-self.classifier_predicts:]:
          batch.append(images[idx])

        preds = self.text_model['multiling'].predict_batch(batch)
        # print(preds)
        ru = [len(re.findall('[а-яА-Я]', el)) > len(re.findall('[a-zA-Z]', el)) for el in preds]
        
        res = np.sum(ru) > len(preds) / 2
        return 'ru' if res else 'en'

    # def predict_language(self, images):
    #     self.classifier_predicts = 19
    #     p = np.argsort([el.shape[1] for el in images])
    #     batch = []
    #     for idx in p[-self.classifier_predicts:]:
    #       batch.append(self.classifier_transforms(image=images[idx])['image'])

    #     batch = torch.stack(batch).to(device)
    #     preds = self.classifier_model(batch)
    #     preds = torch.argmax(preds, dim=1)
    #     # plot_images([torch.moveaxis(el, 0, -1).detach().cpu().numpy() for el in batch[:10]])
    #     # print(preds[:10])
    #     res = preds.sum() > len(preds) / 2
    #     return 'ru' if res else 'en'

    def __call__(self, img, return_only_language=False):
        img = img.copy()
        img = self.transforms(image=img)['image']
        with torch.no_grad():
          output = {'predictions': []}
          contours = self.seg_model(img)
          images = []
          not_none_contours = []
          for contour in contours:
              if contour is not None:
                  crop = crop_img_by_polygon(img, contour)
                  images.append(crop)
                  not_none_contours.append(contour)

          language = self.predict_language(images)
          if return_only_language:
            return language

          predicted_text = []
          for i in range(0, len(images), self.batch_size):
            predicted_text += self.text_model[language].predict_batch(images[i:i + self.batch_size])

          for pred_text, contour in zip(predicted_text, not_none_contours):
            output['predictions'].append({
                              'polygon': [[int(i[0][0] * 2), int(i[0][1] * 2)] for i in contour],
                              # 'polygon': get_polygon_for_answer(contour, crop),
                              'text': pred_text
                            })
        return output


def main():
    pipeline_predictor = PiepleinePredictor(
        segm_model_path=SEGM_MODEL_PATH,
        ru_ocr_model_path=OCR_RU_MODEL_PATH,
        en_ocr_model_path=OCR_EN_MODEL_PATH,
        classifier_model_path=CLASSIFIER_MODEL_PATH,
        multilingual_model_path=OCR_MULTI_MODEL_PATH,
    )
    pred_data = {}
    for img_name in tqdm(os.listdir(TEST_IMAGES_PATH)):
        image = cv2.imread(os.path.join(TEST_IMAGES_PATH, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_data[img_name] = pipeline_predictor(image)

    with open(SAVE_PATH, "w") as f:
        json.dump(pred_data, f)


if __name__ == "__main__":
    main()
