from pathlib import Path
import cProfile, pstats, io
import random
import csv
from qqdm import qqdm, format_str
import pandas as pd
import cv2
import numpy as np
import torch
from utils import misc



# Distillers
class Distiller():
    def __init__(self, teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, device):
        self.teacher = teacher
        self.student = student
        self.processor = processor
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
    
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
    def __init__(self, teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, device):
        self.super().__init__(teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, device)

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
    def __init__(self, teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, loss_weights=[0,0,0,0,1,0], 
                 profile=False, device='cuda'):
        
        super().__init__(teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, device)

        self.edge_filter = False
        self.edge_width = 20
        self.mse_loss = torch.nn.MSELoss(reduction='mean').to(self.device)
        self.focal_loss = NormalizedFocalLossSigmoid().to(self.device) # FocalLoss().to(self.device)
        self.dice_loss = DiceLoss().to(self.device)
        self.bce_loss = BBCEWithLogitLoss().to(self.device)
        self.iou_loss = IoULoss().to(self.device)
        self.FW, self.DW, self.BW, self.IW, self.MW, self.SW, self.KW = loss_weights
        self.pr = cProfile.Profile() if profile else None

    def get_loss(self, t_mask, s_mask, label):
        t_mask_bin = (t_mask > 0.0)
        focal = self.focal_loss(s_mask, t_mask_bin.float())
        bce = self.bce_loss(s_mask, t_mask_bin.float())
        iou = self.iou_loss(s_mask, t_mask_bin.int())
        size = t_mask_bin.float().mean() if self.SW > 0 else 0
        #dice = self.dice_loss(s_mask, t_mask_bin.int())
        #mse = self.mse_loss(torch.relu(s_mask), torch.relu(t_mask))

        bce_gt = self.bce_loss(s_mask, label.float())
        iou_gt = self.iou_loss(s_mask, label.int())

        return focal, bce_gt, bce, iou, iou_gt, size
    
    def get_metrics(self, t_mask, s_mask, label):
        t_mask_bin = (t_mask > 0.0)
        # focal = focal_metric(s_mask, t_mask_bin.float())
        # bce = bce_metric(s_mask, t_mask_bin.float())
        iou = iou_metric(s_mask, t_mask_bin.int())
        iou_gt = iou_metric(s_mask, label.int())
        return iou_gt, iou

    def get_prompt(self, label, seed=None):
        if self.edge_filter:
            e = cv2.Canny(image=label.cpu().numpy().astype(np.uint8), threshold1=10, threshold2=50)
            e = cv2.dilate(e, np.ones((self.edge_width, self.edge_width), np.uint8), iterations = 1).astype(bool)
            label[e] = 0

        C = np.unique(label.cpu())[1:]
        if len(C) == 0:
            c = 0
        else:
            if seed is not None:
                np.random.seed(seed)
            c = np.random.choice(C)
        x_v, y_v = np.where(label.cpu() == c)
        if seed is not None:
            random.seed(seed)
        r = random.randint(0, len(x_v) - 1, )
        x, y = x_v[r], y_v[r]

        return [[[y,x]]], c # inverted to compensate different indexing

    def get_masks(self, img, prompt, t_features):
        # Teacher
        inputs = self.processor(img, input_points=prompt, return_tensors="pt").to(self.device)
        inputs.pop("pixel_values", None) # pixel_values are no more needed
        inputs.update({"image_embeddings": t_features})
        with torch.no_grad():
            outputs = self.teacher(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], 
            inputs["reshaped_input_sizes"], binarize=False)[0]
        scores = outputs.iou_scores
        t_mask = masks.squeeze()[scores.argmax()]
        # Student
        self.student.set_image(img[0].to(self.device))
        masks, scores, _ = self.student.predict(np.array(prompt[0]), np.array([1]), size=t_mask.mean(), return_logits=True)
        s_mask = masks.squeeze()[scores.argmax()]
        return t_mask, s_mask

    def distill(self, epochs=8, accumulate=4, use_saved_features=False, name=''):

        if self.pr:
            self.pr.enable()
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')
            self.student.model.mask_decoder.train()
            self.student.model.prompt_encoder.train()
            self.student.model.image_encoder.eval()

            t = qqdm(self.dataloader, desc=format_str('bold', 'Distillation'))
            r_focal, r_iou_gt, r_bce, r_iou, r_loss, r_bce_gt = 0, 0, 0, 0, 0, 0
            for i, (img, label, _, feats) in enumerate(t):
                feats = feats.to(self.device) if use_saved_features else None
                prompt, c = self.get_prompt(label[0])
                label = label.to(self.device)

                t_mask, s_mask = self.get_masks(img, prompt, feats)
                focal, bce_gt, bce, iou, iou_gt, size = self.get_loss(t_mask, s_mask, label[0]==c)
                loss_kd = self.FW * focal + self.BW * bce + self.IW * iou
                loss_gt = self.BW * bce_gt + self.IW * iou_gt
                loss = (self.KW * loss_kd + loss_gt) * (1-size) / accumulate
                loss.backward(retain_graph=True)

                r_loss += loss.item()
                r_focal += focal.item()
                r_iou_gt += iou_gt.item()
                r_bce += bce.item()
                r_iou += iou.item()
                r_bce_gt += bce_gt.item()
                
                if (i+1) % accumulate == 0 or i+1 == len(self.dataloader):
                    t.set_infos({'Loss':f'{r_loss:.3e}', 'Focal':f'{r_focal/accumulate:.3f}', 
                                 'IoU':f'{r_iou/accumulate:.3f}', 'BCE':f'{r_bce/accumulate:.3f}', 
                                 'IoU GT':f'{r_iou_gt/accumulate:.3f}', 'BCE GT':f'{r_bce_gt/accumulate:.3f}', 
                                 'LR':f'{self.optimizer.param_groups[0]["lr"]:.0e}'}) 
                    r_focal, r_iou_gt, r_bce, r_iou, r_loss, r_bce_gt = 0, 0, 0, 0, 0, 0
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            torch.save(self.student.model.state_dict(), f'bin/distilled_mobile_sam_{name}_{e}.pt')
            self.validate(use_saved_features=use_saved_features)
            self.scheduler.step()

            if self.pr:
                self.pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(self.pr, stream=s).sort_stats('time')
                ps.dump_stats('results/profile.prof')
                ps.print_stats(.1)

    def validate(self, use_saved_features=False):
        self.student.model.mask_decoder.eval()
        self.student.model.prompt_encoder.eval()
        self.student.model.image_encoder.eval()

        with torch.no_grad():
            t = qqdm(self.test_dataloader, desc=format_str('bold', 'Validation'))
            r_bce, r_iou, r_loss, r_iou_m, r_iou_gt_m, r_iou_gt, r_bce_gt = 0, 0, 0, 0, 0, 0, 0
            for i, (img, label, _, feats) in enumerate(t):
                feats = feats.to(self.device) if use_saved_features else None
                prompt, c = self.get_prompt(label[0], seed=0)
                label = label.to(self.device)

                t_mask, s_mask = self.get_masks(img, prompt, feats)
                
                focal, bce_gt, bce, iou, iou_gt, size = self.get_loss(t_mask, s_mask, label[0]==c)
                loss_kd = self.FW * focal + self.BW * bce + self.IW * iou
                loss_gt = self.BW * bce_gt + self.IW * iou_gt
                loss = (self.KW * loss_kd + loss_gt) * (1-size)

                iou_gt_m, iou_m = self.get_metrics(t_mask, s_mask, label[0]==c)
                r_loss += loss.item()
                r_bce +=bce.item()
                r_bce_gt += bce_gt.item()
                r_iou_gt_m += iou_gt_m
                r_iou += iou.item()
                r_iou_gt += iou_gt.item()
                r_iou_m += iou_m
                t.set_infos({'Loss':f'{r_loss/(i+1):.3e}', 'BCE':f'{r_bce/(i+1):.3f}', 'BCE GT':f'{r_bce_gt/(i+1):.3f}', 
                             'IoU Loss':f'{r_iou/(i+1):.3f}', 'IoU Loss GT':f'{r_iou_gt/(i+1):.3f}', 
                             'IoU KD':f'{r_iou_m/(i+1):.3f}', 'IoU GT':f'{r_iou_gt_m/(i+1):.3f}'})



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

'''   Deprecated IoU Loss
class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, target):
        # inputs => N x H x W (logits)
        # target => N x H x W (one-hot)
        N = inputs.size()[0]

        inputs = torch.sigmoid(inputs)

        # Numerator Product
        inter = inputs * target
        ## Sum over all pixels N x H x W => N x C
        inter = inter.view(N, -1).sum(1)

        #Denominator 
        union = inputs + target - (inputs*target)
        ## Sum over all pixels N x H x W => N x C
        union = union.view(N, -1).sum(1)

        loss = -torch.log((inter + 1) / (union + 1))

        ## Return average loss over classes and batch
        return loss.mean()
'''
    
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