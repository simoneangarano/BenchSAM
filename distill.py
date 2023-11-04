from pathlib import Path
from tqdm import tqdm
import pandas as pd
import csv
import cv2
import random

import torch
from transformers import (SamModel, SamProcessor)
from mobile_sam import sam_model_registry

import sys
sys.path.append('..')
from utils.predictor import SamPredictor
from utils.datasets import SA1B_Dataset
from utils.utils import *
from utils.distill_utils import *



class DecDistiller():
    def __init__(self, teacher, student, processor, dataloader, optimizer, scheduler, loss_weights=[0,0,1,0], device='cuda'):
        self.teacher = teacher
        self.student = student
        self.processor = processor
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.edge_filter = True
        self.edge_width = 20
        self.focal_loss = FocalLoss().to(self.device)
        self.dice_loss = DiceLoss().to(self.device)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        self.FW, self.DW, self.BW, self.SW = loss_weights

    def get_distillation_losses(self, t_mask, s_mask):
        t_mask_bin = (t_mask > 0.0)
        focal = self.focal_loss(s_mask, t_mask_bin.float())
        dice = self.dice_loss(s_mask, t_mask_bin.int())
        bce = self.bce_loss(s_mask, torch.sigmoid(t_mask))
        size = t_mask_bin.float().mean() if self.SW > 0 else 0
        return focal, dice, bce, size

    def get_prompt(self, label):
        if self.edge_filter:
            e = cv2.Canny(image=label.cpu().numpy().astype(np.uint8), threshold1=10, threshold2=50)
            e = cv2.dilate(e, np.ones((self.edge_width, self.edge_width), np.uint8), iterations = 1).astype(bool)
            label[e] = 0

        C = np.unique(label.cpu())[1:]
        if len(C) == 0:
            c = 0
        else:
            c = np.random.choice(C)
        x_v, y_v = np.where(label.cpu() == c)
        r = random.randint(0, len(x_v) - 1)
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
        img = img[0]
        self.student.set_image(img)
        masks, scores, _ = self.student.predict(np.array(prompt[0]), np.array([1]), return_logits=True)
        s_mask = masks.squeeze()[scores.argmax()]
        return t_mask, s_mask


    def distill(self, epochs=8, accumulate=4, use_saved_features=False, name=''):
        self.student.model.mask_decoder.train()
        self.student.model.image_encoder.eval()
        for e in range(epochs):
            print(f'Epoch {e+1}/{epochs}')
            t = tqdm(self.dataloader)
            r_focal, r_dice, r_bce, r_loss = 0, 0, 0, 0
            for i, (img, label, _, feats) in enumerate(t):
                img = img.to(self.device)
                label = label.to(self.device)
                feats = feats.to(self.device) if use_saved_features else None

                prompt, _ = self.get_prompt(label[0])
                t_mask, s_mask = self.get_masks(img, prompt, feats)
                focal, dice, bce, size = self.get_distillation_losses(t_mask, s_mask)

                loss = (self.FW * focal + self.DW * dice + self.BW * bce) * (1-size) / accumulate
                loss.backward()

                r_loss += loss.item()
                r_focal += focal.item()
                r_dice += dice.item()
                r_bce += bce.item()
                
                if (i+1) % accumulate == 0 or i+1 == len(self.dataloader):
                    t.set_postfix({'Loss':r_loss/(i+1), 'Focal':r_focal/(i+1), 'Dice':r_dice/(i+1), 
                                   'BCE':r_bce/(i+1), 'LR':self.optimizer.param_groups[0]['lr']})
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    
            self.scheduler.step()
            torch.save(self.student.model.state_dict(), f'bin/distilled_mobile_sam_{name}_{e}.pt')



class EncDistiller():
    def __init__(self, teacher, student, processor, dataloader, optimizer, scheduler, device):
        self.teacher = teacher
        self.student = student
        self.processor = processor
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device


    def get_distillation_loss(self, img, id):
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


    def save_teacher_features(self, path):
        ids, features = [], []
        self.teacher.eval()
        self.teacher.to(self.device)
        t = tqdm(self.dataloader, desc='Distillation')
        for _, (img, _, n) in enumerate(t):
            with torch.no_grad():
                ids.append(n)
                inputs = self.processor(img, input_points=None, return_tensors="pt").to(self.device)
                features.append(self.teacher.get_image_embeddings(inputs["pixel_values"]).cpu())
        features = torch.cat(features, dim=0)
        torch.save(features, path)
        pd.DataFrame(ids).to_csv(path.parent.joinpath('feature_ids.csv'), index=False)


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
            t = tqdm(self.dataloader, desc='Distillation')
            running_loss = 0
            for i, (img, _, id, _) in enumerate(t):
                loss = self.get_distillation_loss(img, id) / accumulate
                running_loss += loss.item()
                t.set_postfix({'Loss':running_loss/(i+1)})
                loss.backward()
                if (i+1) % accumulate == 0 or i+1 == len(self.dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            torch.save(self.student.model.state_dict(), 'bin/distilled_mobile_sam.pt')



def main():

    DATA_DIR = Path('../Datasets/')
    SPLIT = 'sa_000020'
    GPU = 3
    DEVICE = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 8
    SHUFFLE = True
    LOAD_FEATURES = True
    FEATURES = 'results/teacher_features.pt' if LOAD_FEATURES else None

    EPOCHS = 16
    LR = 1e-6
    OPTIM = 'adam'
    WD = 1e-5
    LOSS_WEIGHTS = [1,1,0,0] # 20 focal, 1 dice, 0 bce, 0 size

    MODE = 'decoder' # encoder, decoder, save_features
    PRETRAINED = True if MODE == 'decoder' else False
    EXP = 'fd'

    dataset = SA1B_Dataset(root=DATA_DIR.joinpath('SA_1B/images/'), split=SPLIT,  features=FEATURES, labels=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=SHUFFLE, num_workers=16, pin_memory=True)

    teacher = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
    teacher.eval()
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    model_type = "vit_t"
    sam_checkpoint = "bin/mobile_sam.pt" if PRETRAINED else None

    model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(DEVICE)
    model.eval()
    for m in model.image_encoder.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
    student = SamPredictor(model)

    if MODE == 'encoder':
        DISTILLER = EncDistiller
        params = student.model.image_encoder.parameters()
    else:
        DISTILLER = DecDistiller
        params = student.model.mask_decoder.parameters()

    if OPTIM == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    elif OPTIM == 'adam':
        optimizer = torch.optim.Adam(params, lr=LR)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    distiller = DISTILLER(teacher, student, processor, dataloader, optimizer, scheduler, loss_weights=LOSS_WEIGHTS, device=DEVICE)

    if MODE == 'save_features':
        distiller.save_teacher_features(Path('results/teacher_features.pt'))
    else:
        distiller.distill(epochs=EPOCHS, accumulate=BATCH_SIZE, use_saved_features=LOAD_FEATURES, name=f'{MODE}_{EXP}')



if __name__ == "__main__":
    main()