from pathlib import Path
from sympy import N
from tqdm import tqdm
import pandas as pd
import csv

import torch
from transformers import (SamModel, SamProcessor)
from mobile_sam import sam_model_registry
from predictor import SamPredictor

from datasets import SA1B_Dataset
from utils import *



class Distiller():
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
            for i, (img, _, id) in enumerate(t):
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
    GPU = 2
    DEVICE = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 8
    SHUFFLE = True
    LOAD_FEATURES = True
    FEATURES = 'results/teacher_features.pt'

    PRETRAINED = False
    EPOCHS = 8
    LR = 1e-3
    OPTIM = 'adamw'
    WD = 5e-4

    MODE = 'distill'

    dataset = SA1B_Dataset(root=DATA_DIR.joinpath('SA_1B/images/'), features=FEATURES)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=SHUFFLE, num_workers=8, pin_memory=True)

    teacher = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE).eval()
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    model_type = "vit_t"
    sam_checkpoint = "bin/mobile_sam.pt" if PRETRAINED else None

    model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(DEVICE).train()
    student = SamPredictor(model)

    if OPTIM == 'adamw':
        optimizer = torch.optim.AdamW(student.model.image_encoder.parameters(), lr=LR, weight_decay=WD)
    elif OPTIM == 'adam':
        optimizer = torch.optim.Adam(student.model.image_encoder.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    distiller = Distiller(teacher, student, processor, dataloader, optimizer, scheduler, DEVICE)

    if MODE == 'distill':
        distiller.distill(epochs=EPOCHS, accumulate=BATCH_SIZE, use_saved_features=LOAD_FEATURES)
    elif MODE == 'save_features':
        distiller.save_teacher_features(Path('results/teacher_features.pt'))

if __name__ == "__main__":
    main()