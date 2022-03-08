import cv2
import albumentations as A
import os
import torch
import numpy as np

from albumentations.pytorch.transforms import ToTensor
from transformers import AutoFeatureExtractor, XLMRobertaTokenizer, VisionEncoderDecoderModel, RobertaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlbuPadding(A.DualTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(AlbuPadding, self).__init__(always_apply, p)

    def apply(self, image, **params):
        zeros = np.zeros((128, 384, 3))
        image = np.concatenate([zeros, image, zeros], axis=0)
        return image.astype(np.uint8)


class TrOcrModel:
  def __init__(self, model_path, padding=True):
      self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
      self.model.eval()

      self.feature_extractor = AutoFeatureExtractor.from_pretrained('trocr-small-handwritten-feature-extractor')
      self.tokenizer = XLMRobertaTokenizer.from_pretrained('trocr-small-handwritten-tokenizer')
      
      if padding:
        self.transforms = A.Compose([
                A.Resize(128, 384),
                AlbuPadding(always_apply=True),
            ])
      else:
        self.transforms = A.Compose([
                A.Resize(384, 384),
            ])
  
  def image_preprocess(self, image):
      image = self.transforms(image=image)['image']
      pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
      return pixel_values

  def predict_batch(self, images):
      batch = torch.concat([self.image_preprocess(image) for image in images], axis=0).to(device)
      outputs = self.model.generate(batch)
      return [self.tokenizer.decode(pred.cpu().numpy(), skip_special_tokens=True) for pred in outputs]

  def __call__(self, image):
      pred = self.model.generate(self.image_preprocess(image).to(device))
      return self.tokenizer.decode(pred[0].cpu().numpy(), skip_special_tokens=True)
       
