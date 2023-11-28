from pathlib import Path
import cProfile, pstats, io
import random
import csv
from qqdm import qqdm, format_str
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils import misc



# Distillers
class Distiller():
    def __init__(self, teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, cfg):
        self.teacher = teacher
        self.student = student
        self.processor = processor
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = cfg['DEVICE']
        self.debug = cfg['DEBUG']
        self.cfg = cfg
    
    def save_teacher_features(self, path):
        ids, features = [], []
        self.teacher.eval()
        self.teacher.to(self.device)
        t = qqdm(self.dataloader, desc=format_str('bold', 'Description'))
        for _, (img, _, n) in enumerate(t):
            with torch.no_grad():
                ids.append(n)
                inputs = self.processor(img, input_points=None, return_tensors="pt").to(self.device)
                features.append(self.teacher.get_image_embeddings(inputs["pixel_values"]).cpu())
        features = torch.cat(features, dim=0)
        torch.save(features, path)
        pd.DataFrame(ids).to_csv(path.parent.joinpath('feature_ids.csv'), index=False)

class EncDistiller(Distiller):
    def __init__(self, teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, cfg):
        self.super().__init__(teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, cfg)

    def get_loss(self, img, feats):
        self.student.set_image(img[0].permute((2,0,1)))
        s_features = self.student.features

        if self.use_saved_features:
            idx = self.teacher_features_ids.index(id[0])
            t_features = self.teacher_features[idx].to(self.device).unsqueeze(0)
        else:
            with torch.no_grad():
                inputs = self.processor(img, input_points=None, return_tensors="pt").to(self.device)
                t_features = self.teacher.get_image_embeddings(inputs["pixel_values"])

        return torch.nn.functional.mse_loss(s_features, t_features)

    def distill(self, epochs=8, accumulate=4, use_saved_features=False):
        self.use_saved_features = use_saved_features
        if self.use_saved_features:
            self.teacher_features = torch.load(Path('results/teacher_features.pt'))
            self.teacher_features_ids = csv.reader(open(Path('results/feature_ids.csv'), 'r'))
            self.teacher_features_ids = list(self.teacher_features_ids)
            self.teacher_features_ids = [i[0] for i in self.teacher_features_ids][1:]
            assert len(self.teacher_features) == len(self.teacher_features_ids)

        for e in range(epochs):
            print('Epoch ', e+1)
            t = qqdm(self.dataloader, desc=format_str('bold', 'Description'))
            running_loss = 0
            for i, (img, _, _, feats) in enumerate(t):
                loss = self.get_loss(img, feats) / accumulate
                running_loss += loss.item()
                t.set_postfix({'Loss':running_loss/(i+1)})
                loss.backward()
                if (i+1) % accumulate == 0 or i+1 == len(self.dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            self.scheduler.step()
            torch.save(self.student.model.state_dict(), 'bin/distilled_mobile_sam.pt')

class DecDistiller(Distiller):
    def __init__(self, teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, cfg):
        
        super().__init__(teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, cfg)

        self.edge_filter = cfg['EDGE_FILTER']
        self.edge_width = 20
        self.mse_loss = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.focal_loss = FocalLoss().to(self.device) # FocalLoss().to(self.device)
        self.dice_loss = DiceLoss().to(self.device)
        self.bce_loss = BBCEWithLogitLoss().to(self.device)
        self.iou_loss = IoULoss().to(self.device)
        self.bound_loss = BoundaryLoss().to(self.device)
        self.FW, self.DW, self.BW, self.IW, self.MW, self.SW, self.GW = cfg['LOSS_WEIGHTS']
        self.n_prompts = cfg['PROMPTS']
        self.random_prompt = cfg['RANDOM_PROMPT']
        self.size_thr_low, self.size_thr_high = cfg['SIZE_THRESHOLDS']
        self.pr = cProfile.Profile() if cfg['PROFILE'] else None
        self.best_iou = 0

    def get_features(self, img):
        self.student.set_image(img[0].to(self.device))

    def get_masks(self, img, prompt, t_features):
        # Teacher
        inputs = self.processor(img, input_points=prompt, return_tensors="pt").to(self.device)
        inputs.pop("pixel_values", None) # pixel_values are no more needed
        inputs.update({"image_embeddings": t_features})
        with torch.no_grad():
            outputs = self.teacher(**inputs)
        # Mask
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], 
            inputs["reshaped_input_sizes"], binarize=False)[0]
        scores = outputs.iou_scores
        t_mask = masks.squeeze()[scores.argmax()]
        # Student
        masks, scores, _ = self.student.predict(np.array(prompt[0]), np.array([1]), size=(t_mask>0).float().mean(), return_logits=True)
        s_mask = masks.squeeze()[scores.argmax()]
        return t_mask, s_mask

    def step(self, img, label, feats, acc, use_saved_features=False):
        feats = feats.to(self.device) if use_saved_features else None
        self.get_features(img)
        prompts = self.get_prompts(label[0])
        if prompts == []:
            return False
        label = label.to(self.device)
        for prompt, c in prompts:
            t_mask, s_mask = self.get_masks(img, prompt, feats)
            if self.debug:
                t_mask_bin = (t_mask > 0.0)
                s_mask_bin = (s_mask > 0.0)
            focal, bce, iou, dice, iou_gt = self.get_loss(t_mask, s_mask, label[0]==c)
            loss = (self.FW * focal + self.BW * bce + self.IW * iou + self.DW * dice + self.GW * iou_gt)/(acc*len(prompts))
            loss.backward(retain_graph=True)
            self.update_metrics(loss*acc, focal/len(prompts), bce/len(prompts), iou/len(prompts), dice/len(prompts), iou_gt/len(prompts))
        return True

    def distill(self, name=''):
        self.acc = self.cfg['BATCH_SIZE']
        if self.pr:
            self.pr.enable()
        for e in range(self.cfg['EPOCHS']):
            print(f"Epoch {e+1}/{self.cfg['EPOCHS']} LR {self.optimizer.param_groups[0]['lr']:.0e}")
            self.set_trainable_weights()
            self.t = qqdm(self.dataloader, desc=format_str('bold', 'Distillation'))
            self.init_metrics()
            i = 0
            for img, label, _, feats in self.t:
                check = self.step(img, label, feats, self.cfg['BATCH_SIZE'], self.cfg['LOAD_FEATURES'])
                if check:
                    if (i+1) % self.cfg['BATCH_SIZE'] == 0 or i+1 == len(self.dataloader):
                        self.print_metrics(i+1, self.cfg['BATCH_SIZE'])
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    i += 1

            miou, gtiou = self.validate(use_saved_features=self.cfg['LOAD_FEATURES'])
            iou = gtiou if self.GW == 1 else miou
            if iou > self.best_iou:
                torch.save(self.student.model.state_dict(), f"bin/distilled_mobile_sam_{name}.pt")
                self.best_iou = iou
            self.scheduler.step()

            if self.pr:
                self.pr.disable()
                ps = pstats.Stats(self.pr, stream=io.StringIO()).sort_stats('time')
                ps.dump_stats('results/profile.prof')
                ps.print_stats(.1)

    def validate(self, use_saved_features=False):
        self.set_trainable_weights(train=False)
        with torch.no_grad():
            t = qqdm(self.test_dataloader, desc=format_str('bold', 'Validation'))
            r_bce, r_iou, r_loss, r_focal, r_dice, r_iougt = 0, 0, 0, 0, 0, 0
            i = 0
            for img, label, _, feats in t:
                    feats = feats.to(self.device) if use_saved_features else None
                    self.get_features(img)
                    prompt, c = self.get_prompt(label[0], seed=self.cfg['SEED'])
                    if prompt is None:
                        continue
                    label = label.to(self.device)
                    t_mask, s_mask = self.get_masks(img, prompt, feats)
                    if self.debug:
                        t_mask_bin = (t_mask > 0.0)
                        s_mask_bin = (s_mask > 0.0)
                    focal, bce, iou, dice, iou_gt = self.get_loss(t_mask, s_mask, label[0]==c)
                    loss = self.FW * focal + self.BW * bce + self.IW * iou + self.DW * dice + self.GW * iou_gt
                    if self.cfg['MODEL'] == 'sam':
                        s_mask = t_mask
                    iou, iou_gt = self.get_metrics(t_mask, s_mask, label[0]==c)
                    if self.debug:
                        print(f"mIoU: {iou:.2e}")
                    r_loss += loss.item()
                    r_focal += focal.item()
                    r_bce += bce.item()
                    r_dice += dice.item()
                    r_iou += iou
                    r_iougt += iou_gt
                    t.set_infos({'Loss':f'{r_loss/(i+1):.2e}', 'Focal':f'{r_focal/(i+1):.3f}', 'BCE':f'{r_bce/(i+1):.3f}', 
                                'Dice':f'{r_dice/(i+1):.3f}', 'mIoU':f'{r_iou/(i+1):.3f}', 'mIoUgt':f'{r_iougt/(i+1):.3f}'})
                    i += 1
            return r_iou / (i+1), r_iougt / (i+1)
                
    def get_loss(self, t_mask, s_mask, label):
        t_mask_bin = (t_mask > 0.0)
        focal = self.focal_loss(s_mask, t_mask_bin.float())
        bce = self.bce_loss(s_mask, t_mask_bin.float())
        iou = self.iou_loss(s_mask, t_mask_bin.int())
        dice = self.dice_loss(s_mask, t_mask_bin.int())
        #bce_gt = self.bce_loss(s_mask, label.float())
        iou_gt = self.iou_loss(s_mask, label.int())
        #bound = self.bound_loss(s_mask, label.int())
        if self.debug:
            print(f"BCE: {bce.item():.2e} IoU: {iou.item():.2e} Focal: {focal.item():.2e} Dice: {dice.item():.2e}")
            print(f"IoU_gt: {iou_gt.item():.2e}")

        return focal, bce, iou, dice, iou_gt
    
    def get_metrics(self, t_mask, s_mask, label):
        t_mask_bin = (t_mask > 0.0)
        # focal = focal_metric(s_mask, t_mask_bin.float())
        # bce = bce_metric(s_mask, t_mask_bin.float())
        iou = iou_metric(s_mask, t_mask_bin.int())
        iou_gt = iou_metric(s_mask, label.int())
        return iou, iou_gt

    def get_prompts(self, label, seed=None):
        h, w = label.shape
        margin_h, margin_w = h // self.n_prompts, w // self.n_prompts
        prompts = []
        for point_h in range(self.n_prompts):
            for point_w in range(self.n_prompts):
                crop = label[point_h*margin_h : (point_h+1)*margin_h, point_w*margin_w : (point_w+1)*margin_w]
                local_prompt, c = self.get_prompt(crop, seed=seed)
                if local_prompt is not None:
                    prompt = [[local_prompt[0][0][0] + point_w * margin_w, local_prompt[0][0][1] + point_h * margin_h]]
                    prompts.append(([prompt], c))
        return prompts
    
    def get_prompt(self, label, seed=None):
        if self.edge_filter:
            e = cv2.Canny(image=label.cpu().numpy().astype(np.uint8), threshold1=10, threshold2=50)
            e = cv2.dilate(e, np.ones((self.edge_width, self.edge_width), np.uint8), iterations = 1).astype(bool)
            label[e] = 0
        C, counts = np.unique(label.cpu(), return_counts=True)
        counts = (counts / counts.sum())[1:]
        C = C[1:][(counts >= self.size_thr_low) * (counts <= self.size_thr_high)]
        if len(C) == 0:
            return None, None
        if seed is not None:
            np.random.seed(seed)
        c = np.random.choice(C)

        x_v, y_v = np.where(label.cpu() == c)
        if self.random_prompt:
            if seed is not None:
                random.seed(seed)
            r = random.randint(0, len(x_v) - 1)
            x, y = x_v[r], y_v[r]
        else: # central prompt
            x, y = x_v.mean(), y_v.mean()
            x, y = int(x), int(y)

        return [[[y,x]]], c # inverted to compensate different indexing

    def init_metrics(self):
        self.r_focal, self.r_bce, self.r_iou, self.r_loss, self.r_dice, self.r_iougt = 0, 0, 0, 0, 0, 0

    def update_metrics(self, loss, focal, bce, iou, dice, iou_gt):
        self.r_loss += loss.item()
        self.r_focal += focal.item()
        self.r_dice += dice.item()
        self.r_bce += bce.item()
        self.r_iou += iou.item()
        self.r_iougt += iou_gt.item()

    def print_metrics(self, i, acc):
        self.t.set_infos({'Loss':f'{self.r_loss/(i):.2e}', 'Focal':f'{self.r_focal/(i):.2e}', 
                          'BCE':f'{self.r_bce/(i):.1e}', 'Dice':f'{self.r_dice/(i):.3f}', 
                          'IoU':f'{self.r_iou/(i):.3f}', 'IoU_gt':f'{self.r_iougt/(i):.3f}'}) 
    
    def set_trainable_weights(self, train=True):
        if train:
            if self.cfg['MODE'] == 'decoder':
                self.student.model.mask_decoder.train()
                self.student.model.prompt_encoder.train()
                self.student.model.image_encoder.eval()
            elif self.cfg['MODE'] == 'prompt':
                self.student.model.prompt_encoder.point_embeddings[4].train()
            else:
                raise ValueError(f"Invalid mode: {self.cfg['MODE']}")
        else:
            self.student.model.mask_decoder.eval()
            self.student.model.prompt_encoder.eval()
            self.student.model.image_encoder.eval()



# Loss Functions
class NormalizedFocalLossSigmoid(torch.nn.Module):
    def __init__(self, axis=-1, alpha=0.5, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=False,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss.mean()

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss
   
class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        union = (pred + target).sum() - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)

class BBCEWithLogitLoss(torch.nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = torch.nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

class BoundaryLoss(torch.nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        pred = torch.sigmoid(pred[None, None, ...])
        gt = gt[None, ...]
        bg_pred = torch.ones_like(pred) - pred
        fg_pred = pred.clone()
        pred = torch.cat([bg_pred, fg_pred], dim=1)

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)
        print(pred.shape, one_hot_gt.shape)
        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label

# Metrics
def iou_metric(pred, target):
    pred = pred > 0.0
    target = target > 0.0
    intersection = (pred & target).float().sum((0,1))
    union = (pred | target).float().sum((0,1))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def dice_metric(pred, target):
    pred = pred > 0.0
    target = target > 0.0
    intersection = (pred & target).float().sum((0,1))
    union = (pred | target).float().sum((0,1))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

def bce_metric(pred, target):
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, target).item()

def focal_metric(pred, target):
    return FocalLoss()(pred, target).item()