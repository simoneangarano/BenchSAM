import sys, logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pycocotools

### Metrics ###

def calculate_metrics(target, pred, eps=1e-5, verbose=False):
    output = pred.view(-1)
    target = target.view(-1)

    tp = torch.sum(output * target)  # TP (Intersection)
    un = torch.sum(output + target)  # Union
    fp = torch.sum(output * (~target))  # FP
    fn = torch.sum((~output) * target)  # FN
    tn = torch.sum((~output) * (~target))  # TN

    iou = (tp + eps) / (un + eps)
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    if verbose:
        print(f"IoU: {iou:.4f}, Pixel Acc: {pixel_acc:.4f}, Dice: {dice:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}")

    return iou, pixel_acc, dice, precision, specificity, recall



### Visualization ###
def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is not None:
        color = np.array(color)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(6,6))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def get_mask_limits(masks):
  if len(masks) == 0:
    return np.zeros((0, 4), dtype=int)
  n = len(masks)
  bb = np.zeros((n, 4), dtype=int)
  for index, mask in enumerate(masks):
      if not mask.any():
          continue
      y, x = np.where(mask != 0)
      bb[index, 0] = np.min(x)
      bb[index, 1] = np.min(y)
      bb[index, 2] = np.max(x)
      bb[index, 3] = np.max(y)
      mask = mask[bb[index, 1]:bb[index, 3], bb[index, 0]:bb[index, 2]]

  return np.min(bb[:,:2], axis=0).tolist(), np.max(bb[:,2:], axis=0).tolist(), mask

C = [[0.00, 0.65, 0.88, 0.6],[0.95, 0.47, 0.13, 0.6]]

def show_points_and_masks_on_image(raw_image, masks, input_points, input_labels=None, zoom=True):
    plt.figure(figsize=(5,5))
    plt.imshow(raw_image, alpha=0.6)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca()) 
    for i, m in enumerate(masks):
      show_mask(m, plt.gca(), color=C[i])
    if zoom:
      min, max, _ = get_mask_limits(masks)
      plt.xlim(min[0], max[0])
      plt.ylim(max[1], min[1])
    plt.axis('off')
    plt.show()

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='none', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='none', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)

def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()



# Data helper functions
def get_dataset_info(dataset):
    if dataset == 'coco':
        root = Path("../Datasets/coco-2017/val2017/")
        n = 92
        classes = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                   'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'kite', 
                   'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 
                   'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                   'hair drier', 'toothbrush', 'hair brush']
    elif dataset == 'cityscapes':
        root = Path("../Datasets/Cityscapes/leftImg8bit/val/")
        n = 19
        classes = ['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic Light', 'Traffic Sign', 'Vegetation', 
                   'Terrain', 'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle']
    elif dataset == 'sa1b':
        root = Path("../Datasets/SA_1B/images/")
        n = 0
        classes = []
    
    return root, n, classes

def annToRLE(im, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    h, w = im.shape[2:]
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = pycocotools.mask.frPyObjects(segm, h, w)
        rle = pycocotools.mask.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = pycocotools.mask.frPyObjects(segm, h, w)
    else:
        # rle
        print('RLE')
        rle = ann['segmentation']
    return rle

def annToMask(im, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(im, ann)
    m = pycocotools.mask.decode(rle)
    return m
    
# Suppress outputs
import contextlib
import io
import sys

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

# GPU