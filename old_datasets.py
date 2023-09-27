import os
import os.path as osp

import random
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

from PIL import Image, ImageOps, ImageFilter

__all__ = ['SegmentationDataset']

class CityScapes_Dataset(torchvision.datasets.Cityscapes):
    def __init__(self, data_dir, label_dir=None, split='train', n_TTA=0):
        super(CityScapes_Dataset, self).__init__(root=data_dir, split=split, mode='fine', target_type='semantic', 
                                                 transform=torchvision.transforms.ToTensor(),
                                                 target_transform=torchvision.transforms.ToTensor(),
                                                 transforms=None)
        self.ignore_label = 255

        self.CLASSES = [#'Unlabeled', 0
                        #'Ego vehicle', 1
                        #'Rectification border', 2
                        #'Out of roi','static', 3
                        #'Dynamic', 4
                        #'Ground', 5
                        #'Road', 6
                        'Sidewalk', #7
                        'Parking', #8
                        #'Rail track', 9
                        #'Building', 10
                        'Wall', #11
                        'Fence', #12
                        'Guard rail', #13
                        #'Bridge', 14
                        #'Tunnel', 15
                        #'Pole', 16
                        'Polegroup', #17
                        #'Traffic light', 18
                        'Traffic sign', #19
                        'Vegetation', #20
                        'Terrain', #21
                        'Sky', #22
                        'Person', #23
                        'Rider', #24
                        'Car', #25
                        'Truck', #26
                        'Bus', #27
                        'Caravan', #28
                        #'Trailer', 29
                        #'Train', 30
                        'Motorcycle', #31
                        'Bicycle', #32
                        'License plate' #33
        ]

        self.cls_num_list = None

# Manually-defined CityScapes dataset
class CityScapes_Dataset_Manual(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, n_TTA=0):
        self.data_dir = data_dir
        self.split = split
        self.n_TTA = n_TTA
        self.ignore_label = 255

        self.CLASSES = ['Unlabeled',
                        'Ego vehicle',
                        'Rectification border',
                        'Out of roi','static',
                        'Dynamic',
                        'Ground',
                        'Road',
                        'Sidewalk',
                        'Parking',
                        'Rail track',
                        'Building',
                        'Wall',
                        'Fence',
                        'Guard rail',
                        'Bridge',
                        'Tunnel',
                        'Pole',
                        'Polegroup',
                        'Traffic light',
                        'Traffic sign',
                        'Vegetation',
                        'Terrain',
                        'Sky',
                        'Person',
                        'Rider',
                        'Car',
                        'Truck',
                        'Bus',
                        'Caravan',
                        'Trailer',
                        'Train',
                        'Motorcycle',
                        'Bicycle',
                        'License plate'
        ]

        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                              3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                              7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                              14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                              18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 29: self.ignore_label, 30: self.ignore_label, 31: 16,
                              32: 17, 33: 18}

        self.cls_num_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

        self.img_paths = [image_id.strip() for image_id in open(data_dir)]

        self.files = []
        for item in self.img_paths:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({"img": img_file,
                               "label": label_file,
                               "name": name})

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

        self.tta_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def generate_scale_label(self, image, label):
        f_scale = 0.7 + random.randint(0, 14) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])

        if self.split == 'test':
            x = cv2.resize(x, (256, 256))
        else:
            x = cv2.resize(x, (256, 256))

        if self.n_TTA > 0:
            x = torch.stack([self.tta_transform(x) for _ in range(self.n_TTA)], dim=0)
        else:
            x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).float()

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = self.id2trainId(label)
        size = image.shape
        name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(), label.copy(), np.array(size), name

class NIH_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, n_TTA=0):
        self.data_dir = data_dir
        self.split = split
        self.n_TTA = n_TTA

        self.CLASSES = [
            'No Finding',  # shared
            'Infiltration',
            'Effusion',  # shared
            'Atelectasis',  # shared
            'Nodule',
            'Mass',
            'Consolidation',  # shared
            'Pneumothorax',  # shared
            'Pleural Thickening',
            'Cardiomegaly',  # shared
            'Emphysema',
            'Edema',  # shared
            'Fibrosis',
            'Subcutaneous Emphysema',  # shared
            'Pneumonia',  # shared
            'Tortuous Aorta',  # shared
            'Calcification of the Aorta',  # shared
            'Pneumoperitoneum',  # shared
            'Hernia',
            'Pneumomediastinum'  # shared
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'miccai2023_nih-cxr-lt_labels_{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

        self.tta_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])

        if self.split == 'test':
            x = cv2.resize(x, (256, 256))
        else:
            x = cv2.resize(x, (256, 256))

        if self.n_TTA > 0:
            x = torch.stack([self.tta_transform(x) for _ in range(self.n_TTA)], dim=0)
        else:
            x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).float()

class MIMIC_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, n_TTA=0):
        self.split = split
        self.n_TTA = n_TTA

        self.CLASSES = [
            'Support Devices',
            'Lung Opacity',
            'Cardiomegaly',  # shared
            'Pleural Effusion',  # shared
            'Atelectasis',  # shared
            'Pneumonia',  # shared
            'Edema',  # shared
            'No Finding',  # shared
            'Enlarged Cardiomediastinum',
            'Consolidation',  # shared
            'Pneumothorax',  # shared
            'Fracture',
            'Calcification of the Aorta',  # shared
            'Tortuous Aorta',  # shared
            'Subcutaneous Emphysema',  # shared
            'Lung Lesion',
            'Pneumomediastinum',  # shared
            'Pneumoperitoneum',  # shared
            'Pleural Other'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'miccai2023_mimic-cxr-lt_labels_{split}.csv'))

        self.img_paths = self.label_df['path'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        # self.img_paths = [f[:-4] + '_320.jpg' for f in self.img_paths]  # path to version of image pre-downsampled to 320x320
        self.labels = self.label_df[self.CLASSES].values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

        self.tta_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])

        if self.split == 'test':
            x = cv2.resize(x, (256, 256))
        else:
            x = cv2.resize(x, (256, 256))

        if self.n_TTA > 0:
            x = torch.stack([self.tta_transform(x) for _ in range(self.n_TTA)], dim=0)
        else:
            x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).float()


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


class CitySegmentation(SegmentationDataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19

    def __init__(self, root='../datasets/citys', split='train', mode=None, transform=None, **kwargs):
        super(CitySegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(self.root), "Please setup the dataset using ../datasets/cityscapes.py"
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/' + split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths
