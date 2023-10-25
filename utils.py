from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import pycocotools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple

### Constants ###

METRICS = ['iou', 'mask_size_diff', 'score_diff', 'precision', 'recall']
MODELS = ['SAM', 'FastSAM', 'MobileSAM']
SPARSITIES = [10, 20, 30, 40, 50, 60, 70, 80, 90]
MODES = ['', '_gup']
PARAMS = {'SAM': 615e6,
          'FastSAM': 68e6,
          'MobileSAM': 9.66e6}
SAM = {'iou': 1.0,
       'mask_size_diff': 0.0,
       'score_diff': 0.0,
       'precision': 1.0,
       'recall': 1.0}
C = [[0.00, 0.65, 0.88, 0.6],[0.95, 0.47, 0.13, 0.6]]



### Metrics ###

def get_metrics(target, pred, eps=1e-5, verbose=False):

    if verbose:
        plt.subplot(1, 2, 1)
        plt.imshow(target)
        plt.subplot(1, 2, 2)
        plt.imshow(pred)
        plt.show()

    output = np.reshape(pred, -1)
    target = np.reshape(target, -1)

    tp = np.sum(output * target)  # TP (Intersection)
    un = np.sum(output + target)  # Union
    fp = np.sum(output * (~target))  # FP
    fn = np.sum((~output) * target)  # FN
    tn = np.sum((~output) * (~target))  # TN

    iou = (tp + eps) / (un + eps)
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    if verbose:
        print(f"IoU: {iou:.4f}, Pixel Acc: {pixel_acc:.4f}, Dice: {dice:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}")

    return iou, pixel_acc, dice, precision, specificity, recall

def get_analytics(target_df, pred_df, prompt_df, cfg):
    metrics = {k: [] for k in ['name', 'prompt', 'class', 't_class', 's_class', 'score', 'score_diff', 'mask_size', 
                               'mask_size_diff', 'iou', 'pixel_acc', 'dice', 'precision', 'recall', 'specificity']}
    for i in range(len(target_df)):
        target = target_df.loc[i]
        pred = pred_df.loc[i]
        prompt = prompt_df.loc[i]

        if cfg['DATASET'] == 'sa1b':
            t = get_full_mask(target['mask'], target['mask_origin'], prompt['shape'])
            p = get_full_mask(pred['mask'], pred['mask_origin'], prompt['shape'])
        else:
            t = target['mask']
            p = pred['mask']

        iou, pixel_acc, dice, precision, specificity, recall = get_metrics(t.astype(bool), p.astype(bool))
        
        metrics['name'].append(target['name'])
        metrics['prompt'].append(target['prompt'])
        metrics['class'].append(target['class'])
        metrics['t_class'].append(target['s_class'])
        metrics['s_class'].append(pred['s_class'])
        metrics['score'].append(pred['score'])
        metrics['score_diff'].append((pred['score'] - target['score']) / (target['score'] + 1e-5))
        p_size = np.mean(pred['mask'].astype('float'))
    
        t_size = np.mean(target['mask'].astype('float'))
        metrics['mask_size'].append(p_size)
        metrics['mask_size_diff'].append((p_size - t_size) / (t_size + 1e-3))
        metrics['iou'].append(iou)
        metrics['pixel_acc'].append(pixel_acc)
        metrics['dice'].append(dice)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['specificity'].append(specificity)
    
    return pd.DataFrame(metrics)

def save_analytics(cfg):
    df_p = pd.read_pickle(f"results/{cfg['EXPERIMENT']}{cfg['DATASET']}_prompts.pkl")
    df_0 = pd.read_pickle(f"results/{cfg['EXPERIMENT']}{cfg['DATASET']}_{cfg['TARGET']}_0.pkl")
    df_s = pd.read_pickle(f"results/{cfg['EXPERIMENT']}{cfg['DATASET']}_{cfg['MODEL']}_{cfg['SPARSITY']}{cfg['MODE']}.pkl")

    df_0s = get_analytics(df_0, df_s, df_p, cfg)
    df_0s.to_pickle(f"analytics/{cfg['EXPERIMENT']}{cfg['DATASET']}_{cfg['MODEL']}_{cfg['SPARSITY']}{cfg['MODE']}.pkl")
    df_0s.head()

def get_summary(cfg, classwise=False):
    df = pd.read_pickle(f"analytics/{cfg['EXPERIMENT']}{cfg['DATASET']}_{cfg['MODEL']}_{cfg['SPARSITY']}{cfg['MODE']}.pkl")
    df = add_superclass(df, cfg) if classwise else df
    classes = cfg['SUP_N'].values() if classwise else [-1]
    sums = []
    for c in classes:
        df_c = df[df['superclass']==c] if classwise else df
        summary = {'id': cfg['SPARSITY'] if cfg['SPARSITY'] != 0 else cfg['MODEL'],
                   'pruning': cfg['MODE'], 'class': c, 'n': len(df_c)}
        for k in ['iou', 'mask_size_diff', 'score_diff', 'precision', 'recall']:
            summary[k] = df_c[k].mean()
            summary[k+'_std'] = df_c[k].std()
            summary[k+'_hist'] = np.histogram(df_c[k], bins='sturges')
        sums.append(summary)
    return pd.DataFrame(sums)



### Visualization ###

def show_mask(mask, ax, random_color=False, color=None, alpha=0.6):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is not None:
        color = np.array(color)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, interpolation='nearest')

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

def show_points_and_masks_on_image(raw_image, masks, input_points, input_labels=None, zoom=True, prompt_zoom=False, thr=5, save=None):
    plt.figure(figsize=(5,5))
    plt.imshow(raw_image, alpha=0.6)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca()) 
    for i, m in enumerate(masks):
        show_mask(m, plt.gca(), color=C[i], alpha=0.8)
    if prompt_zoom:
        plt.xlim(input_points[:,0]-thr, input_points[:,0]+thr)
        plt.ylim(input_points[:,1]+thr, input_points[:,1]-thr)
    elif zoom:
        min, max, _ = get_mask_limits(masks)
        plt.xlim(min[0], max[0])
        plt.ylim(max[1], min[1])
    plt.axis('off')
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
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

def show_entry(row, target_df, pred_df, cfg, zoom=True, prompt_zoom=False, save=None):
    image = get_image(row['name'], cfg)
    target_mask = target_df[target_df['name']==row['name']]['mask'].values[0]
    pred_mask = pred_df[pred_df['name']==row['name']]['mask'].values[0]
    if cfg['DATASET'] == 'sa1b':
        pred_mask = get_full_mask(pred_mask, pred_df[pred_df['name']==row['name']]['mask_origin'].values[0], image.shape[:2])
        target_mask = get_full_mask(target_mask, target_df[target_df['name']==row['name']]['mask_origin'].values[0], image.shape[:2])

    show_points_and_masks_on_image(image, [pred_mask, target_mask], [row['prompt']], zoom=zoom, prompt_zoom=prompt_zoom, save=save)
    print(f'ID: {row["name"]}, PromptClass: {get_labels(row["class"], cfg)},' 
          'TargetClass: {get_labels(row["t_class"], cfg)}, PredClass: {get_labels(row["s_class"], cfg)},') 
    print(f'ScoreDiff: {row["score_diff"]:.4f}, MaskSizeDiff: {row["mask_size_diff"]:.4f}, IoU: {row["iou"]:.4f}, '
          f'Precision: {row["precision"]:.4f}, Recall: {row["recall"]:.4f}')
    
def show_samples(pie_df, target_df, pred_df, cfg, n=5, zoom=True, prompt_zoom=False, random=False, save=False):
    print('Legend: Target -> Orange, Prediction -> Blue')
    pie_df = pie_df.sample(n, random_state=0) if random else pie_df[:n]
    if save:
        def save(x):
            x -= 1
            return f"figures/{cfg['EXPERIMENT']}{cfg['DATASET']}_{cfg['MODEL']}_{cfg['SPARSITY']}{cfg['MODE']}_{x}.pdf"
    else:
        save = lambda x: None
    pie_df.apply(lambda x: show_entry(x, target_df, pred_df, cfg, zoom=zoom, prompt_zoom=prompt_zoom, save=save(n)), axis=1)

def get_hists(summary, cfg, save=True, plot=True):
    if len(summary) == 9:
        fig, axs = plt.subplots(3, 3, sharex=True, sharey='row', figsize=(10,8))
        title = "% Sparsity"
        j = "id"
    elif len(summary) == 2:
        fig, axs = plt.subplots(1, 2, sharex=True, sharey='row', figsize=(10,3))
        title = ""
        j = "id"
    elif len(summary) == 12:
        fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,8))
        title = " Class"
        j = "class"
    for (axs, i) in zip(axs.flat, range(len(summary))):
        y, x = summary.iloc[i][f"{cfg['METRIC']}_hist"]
        axs.bar(x[:-1], y, width=x[1]-x[0], alpha=1)
        axs.set_title(f"{summary.iloc[i][j]}{title}", fontsize=10)
        axs.semilogy()
    plt.suptitle(f"{cfg['METRIC'].upper()} Distribution", fontsize=12)
    if save:
        plt.savefig(f"figures/{cfg['EXPERIMENT']}{cfg['DATASET']}_{cfg['MODEL']}{cfg['MODE']}_{cfg['METRIC']}.pdf", bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.clf()

def get_curves(s, cfg, std=False, plot=True, save=False):
    gup = s[s['pruning']=='_gup']
    fsam = s[s['id']=='FastSAM']
    msam = s[s['id']=='MobileSAM']
    sgpt = s[s['id'].str.isalpha().isnull()]
    sgpt = sgpt[sgpt['pruning']!='_gup']

    _, ax = plt.subplots()
    ax.grid()
    ax.title.set_text(cfg['METRIC'].replace('_', ' ').capitalize())
    ax.errorbar(PARAMS['SAM'], SAM[cfg['METRIC']], fmt='o', label='SAM')

    _, c, b = ax.errorbar((1 - gup['id']/100) * PARAMS['SAM'], gup[cfg['METRIC']], 
                          yerr=gup[cfg['METRIC']+'_std'] if std else None, xerr=None,
                          fmt='-o', capsize=2, capthick=2, label='Unstructured')
    [bar.set_alpha(0.5) for bar in b]
    [cap.set_alpha(0.5) for cap in c]
    _, c, b = ax.errorbar((1 - sgpt['id']/100) * PARAMS['SAM'], sgpt[cfg['METRIC']], 
                          yerr=sgpt[cfg['METRIC']+'_std'] if std else None, xerr=None,
                          fmt='-o', capsize=2, capthick=2, label='SparseGPT')
    [bar.set_alpha(0.5) for bar in b]
    [cap.set_alpha(0.5) for cap in c]
    _, c, b = ax.errorbar(PARAMS['FastSAM'], fsam[cfg['METRIC']], 
                          yerr=fsam[cfg['METRIC']+'_std'] if std else None, xerr=None,
                          fmt='o', capsize=2, capthick=2, label='FastSAM')
    [bar.set_alpha(0.5) for bar in b]
    [cap.set_alpha(0.5) for cap in c]
    _, c, b = ax.errorbar(PARAMS['MobileSAM'], msam[cfg['METRIC']], 
                          yerr=msam[cfg['METRIC']+'_std'] if std else None, xerr=None,
                          fmt='o', capsize=2, capthick=2, label='MobileSAM')
    [bar.set_alpha(0.5) for bar in b]
    [cap.set_alpha(0.5) for cap in c]
    plt.legend(loc='best')
    plt.xlabel('Parameters')
    plt.ylabel(cfg['METRIC'].replace('_', ' ').capitalize())
    plt.axvline(PARAMS['SAM'], linestyle='--')
    plt.axhline(SAM[cfg['METRIC']], linestyle='--')
    plt.semilogx()
    if save:
        plt.savefig(f"figures/{cfg['EXPERIMENT']}{cfg['DATASET']}_{cfg['METRIC']}_curves.pdf", bbox_inches='tight')
    if plot:
        plt.show()

def get_clusters(data, cfg, plot=False, save=False):
    (n_samples, n_features), n = data.shape, 3
    print(f"#: {n}; # samples: {n_samples}; # features {n_features}")

    data_scaled = StandardScaler().fit_transform(data)
    reduced_data = PCA(n_components=2).fit_transform(data_scaled)
    kmeans = KMeans(init="k-means++", n_clusters=n, n_init=4)
    kmeans.fit(reduced_data)

    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.colormaps["Reds"],
        aspect="auto",
        origin="lower",
        alpha=0.3
    )
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data[:,0])
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=50,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.colorbar()
    if save:
        plt.savefig(f"figures/{cfg['EXPERIMENT']}{cfg['DATASET']}_{cfg['MODEL']}_{cfg['SPARSITY']}{cfg['MODE']}_kmeans.pdf", bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.clf()
    return kmeans.labels_



### Data Helpers ###

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
        superclasses = ('background', 
                        'person', 
                        'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 
                        'outdoor', 'outdoor', 'outdoor', 'outdoor', 'outdoor', 'outdoor', 
                        'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 
                        'accessory', 'accessory', 'accessory', 'accessory', 'accessory', 'accessory', 'accessory', 'accessory', 
                        'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 
                        'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 
                        'food', 'food', 'food', 'food', 'food', 'food', 'food', 'food', 'food', 'food', 
                        'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 
                        'electronic', 'electronic', 'electronic', 'electronic', 'electronic', 'electronic', 
                        'appliance', 'appliance', 'appliance', 'appliance', 'appliance', 'appliance',
                        'indoor', 'indoor', 'indoor', 'indoor', 'indoor', 'indoor', 'indoor', 'indoor')
        superclasses_n = {'background': 0,
                          'person': 1,
                          'vehicle': 2,
                          'outdoor': 3,
                          'animal': 4,
                          'accessory': 5,
                          'sports': 6,
                          'kitchen': 7,
                          'food': 8,
                          'furniture': 9,
                          'electronic': 10,
                          'appliance': 11,
                          'indoor': 12}
        
    elif dataset == 'cityscapes':
        root = Path("../Datasets/Cityscapes/leftImg8bit/val/")
        n = 19
        classes = ['Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 'Traffic Light', 'Traffic Sign', 'Vegetation', 
                   'Terrain', 'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle']
        superclasses = classes
        superclasses_n = {c: i for i, c in enumerate(classes)}

    elif dataset == 'sa1b':
        root = Path("../Datasets/SA_1B/images/")
        n = 0
        classes = []
        superclasses = []
        superclasses_n = {}
    
    return root, n, classes, superclasses, superclasses_n

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
    


### Utils ###

def get_superclass(c, cfg):
    return cfg['SUP_N'][cfg['SUPERCLASSES'][c]]

def add_superclass(df, cfg):
    if cfg['DATASET'] == 'coco':
        df['superclass'] = df['class'].apply(lambda x: get_superclass(x, cfg))
    else:
        df['superclass'] = df['class']
    return df

def get_full_mask(mask, origin, full_shape):
    full_mask = np.zeros(full_shape)
    full_mask[origin[0]:origin[0]+mask.shape[0], origin[1]:origin[1]+mask.shape[1]] = mask
    return full_mask

def add_image_shape(df, cfg):
    df['shape'] = df['name'].apply(lambda x: get_image(x, cfg).shape[:2])
    return df

def get_labels(name, cfg):
    if name is None:
        return 'None'
    elif isinstance(name, list):
        return [get_labels(n, cfg) for n in name]
    else: 
        return cfg['CLASSES'][name].title()

def get_image(name, cfg):
    if cfg['DATASET'] == 'coco':
        image_path = cfg['ROOT'].joinpath(f'{str(name).zfill(12)}.jpg')
    elif cfg['DATASET'] == 'cityscapes':
        image_path = cfg['ROOT'].joinpath(f"{name.split('_')[0]}/{name}")
    elif cfg['DATASET'] == 'sa1b':
        image_path = cfg['ROOT'].joinpath(f"0/{name}.jpg")
    return np.array(Image.open(image_path).convert("RGB"))

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


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)