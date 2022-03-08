import sys
import os
import torch
import warnings
import numpy as np
from albumentations.pytorch.transforms import ToTensor
import albumentations as A
from torch import nn
import collections
import time
import cv2
from numba import jit
from numba.typed import Dict

warnings.filterwarnings("ignore")
IMG_SIZE = 256
cnt_colors = 1
c2c = {}


def get_pad_shape(shape, frame_size = 1024):
  a = (shape[0] + frame_size - 1) // frame_size * frame_size
  b = (shape[1] + frame_size - 1) // frame_size * frame_size
  return (a, b, shape[2])


def get_larger_contour(contours):
    larger_area = 0
    larger_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > larger_area:
            larger_contour = contour
            larger_area = area
    return larger_contour


def get_contours_from_mask(mask, min_area=100):
    # print(mask.shape, mask.dtype)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE
                                           )
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list

@jit(nopython=True)
def dfs(mask, cmp_number, i, j, now_cmp):
  if cmp_number[i, j] != -1:
    return 0
  cmp_number[i, j] = now_cmp

  ans = 1
  n = mask.shape[0]
  m = mask.shape[1]
  for di in range(-1, 2):
    for dj in range(-1, 2):
      if i + di >= 0 and i + di < n and j + dj >= 0 and j + dj < m and mask[i + di, j + dj]:
        ans += dfs(mask, cmp_number, i + di, j + dj, now_cmp)

  return ans
  


@jit(nopython=True)
def leave_biggest_contour(mask, min_area=100):
  cmp_number = np.zeros(mask.shape[:2]).astype(np.int64)
  cmp_number[:, :] = -1
  cnt = Dict()
  now_cmp = 1
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      if mask[i, j] and cmp_number[i, j] == -1:
        cnt[now_cmp] = dfs(mask, cmp_number, i, j, now_cmp)
        now_cmp += 1
        
  max_size = min_area
  biggest_id = -2
  for k, v in cnt.items():
    if max_size < v:
      max_size = v
      biggest_id = k

  return cmp_number == biggest_id

def clear_masks(masks, cnt_colors):
  res = np.zeros((IMG_SIZE, IMG_SIZE)).astype(np.int64)

  was = cnt_colors
  for i in range(len(masks)):
    # masks[i] = leave_biggest_contour(masks[i])
    res[masks[i]] = cnt_colors
    cnt_colors += 1
    
  # for color in range(was, cnt_colors):
  #   mask = (res == color)
  #   res[leave_biggest_contour(mask) ^ mask] = 0
    
  return res, cnt_colors

@jit(nopython=True)
def fix_colors(res, map_color):
    mask_border = np.zeros(res.shape[:2]).astype(np.bool_)
    for i in range(res.shape[0]):
      for j in range(res.shape[1]):
        res[i, j] = map_color[res[i, j]]
        
    for i in range(1, res.shape[0] - 1):
      for j in range(1, res.shape[1] - 1):
        mask_border[i, j] = res[i, j] != res[i - 1, j] or res[i, j] != res[i, j - 1] or \
                            res[i, j] != res[i + 1, j] or res[i, j] != res[i, j + 1]
    
  
    for i in range(res.shape[0]):
      for j in range(res.shape[1]):
        if mask_border[i, j]:
          res[i, j] = 0
    return res

@jit(nopython=True)
def update_gr(gr, a, b):
  for el in zip(a, b):
    if el[0] != 0:
      gr[el[1], el[0]] += 1

def get_map_color(res, check_i, check_j):
    c2c = collections.defaultdict(list)
    # print(res.shape)
    start = time.time()
    gr = np.array([np.zeros(cnt_colors).astype(np.int64) for i in range(cnt_colors)])
    for j in check_j:
      mask = (res[:, j-1] != res[:, j])
      a = res[mask, j-1]
      b = res[mask, j]
      update_gr(gr, a, b)

    for i in check_i:
      mask = (res[i-1, :] != res[i, :])
      a = res[i - 1, mask]
      b = res[i, mask]
      update_gr(gr, a, b)

    # print('time 1: {}'.format(time.time() - start))
    start = time.time()
    map_color = {}

    for i in range(1, cnt_colors):
      if gr[i].max() > 0:
        map_color[i] = np.argmax(gr[i])
    
    for k, v in map_color.items():
      if v in map_color.keys():
        map_color[k] = map_color[v]

    for c in np.unique(res):
      if c not in map_color.keys():
        map_color[c] = c

    d = Dict()
    for k, v in map_color.items():
        d[k] = v
    return d
    # print('time 2: {}'.format(time.time() - start))

def predict_full_img(image, model, device):
    np.random.seed(42)
    global cnt_colors
    cnt_colors = 1
    check_i = []
    check_j = []
    model.eval()
    shift = 50
    frame_size = 512
    batch_size = 16
    image_shape = image.shape
    image_pad = np.zeros((image_shape[0] + frame_size, image_shape[1] + frame_size, 3)).astype(np.uint8)
    image_pad[:image_shape[0], :image_shape[1]] = image
    pad_shape = image_pad.shape
    
    
    res = np.zeros(pad_shape[:2]).astype(np.int)
    alb_transforms = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensor()])
    
    images = []
    for i in range(0, pad_shape[0], frame_size - 2 * shift):
        for j in range(0, pad_shape[1], frame_size - 2 * shift):
            now = image_pad[i:i + frame_size, j:j + frame_size]
            if now.shape != (frame_size, frame_size, 3):
                continue

            now = alb_transforms(image=now)['image'].to(device)
            images.append((now, i, j))

    for ind in range(0, len(images), batch_size):
      batch = [el[0] for el in images[ind: ind + batch_size]]
      i_ind = [el[1] for el in images[ind: ind + batch_size]]
      j_ind = [el[2] for el in images[ind: ind + batch_size]]

      prediction = model(batch)
      for idx in range(len(batch)):
        masks = prediction[idx]['masks']
        masks = masks[prediction[idx]['scores'] > 0.5]
        
        if len(masks) == 0:
          continue
        masks = masks.detach().cpu().numpy() > 0.5
        masks = masks.reshape(-1, 256, 256)
        p = np.argsort(masks.sum(axis=(1, 2)))
        
        masks = masks[p]
        pred, cnt_colors = clear_masks(masks, cnt_colors)
        pred = A.Resize(frame_size, frame_size, interpolation=cv2.INTER_NEAREST)(image=pred)['image']
        i, j = i_ind[idx], j_ind[idx]

        check_i.append(i + shift)
        check_j.append(j + shift)
        res[i + shift:i + frame_size, j + shift:j + frame_size] = pred[shift:, shift:]
      
    check_j = np.unique(check_j)
    check_i = np.unique(check_i)
    map_color = get_map_color(res, check_i, check_j)
    start = time.time()
    
    res = fix_colors(res, map_color)
    return res[:image_shape[0], :image_shape[1]]

