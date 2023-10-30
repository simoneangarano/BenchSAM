from pathlib import Path
from tqdm import tqdm

import torch
from transformers import (SamModel, SamProcessor)
from mobile_sam import sam_model_registry
from predictor import SamPredictor

from datasets import SA1B_Dataset
from utils import *



class Distiller():
    def __init__(self, teacher, student, processor, dataloader, optimizer, device):
        self.teacher = teacher
        self.student = student
        self.processor = processor
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device

    def get_distillation_loss(self, img):
        self.student.set_image(img[0].permute((2,0,1)))
        s_features = self.student.features

        with torch.no_grad():
            inputs = self.processor(img, input_points=None, return_tensors="pt").to(self.device)
            t_features = self.teacher.get_image_embeddings(inputs["pixel_values"])

        return torch.nn.functional.mse_loss(s_features, t_features)

    def distill(self, epochs=5, batch_size=4):
        for _ in range(epochs):
            t = tqdm(self.dataloader, desc='Distillation')
            for i, (img, _, _) in enumerate(t):
                loss = self.get_distillation_loss(img) / batch_size
                loss.backward()
                if (i+1) % batch_size == 0 or i+1 == len(self.dataloader):
                    self.optimizer.step()
                    t.set_postfix({'Loss':loss.item()})
                    self.optimizer.zero_grad()




def main():
    DATA_DIR = Path('../Datasets/')
    PRETRAINED = False
    GPU = 1
    DEVICE = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    dataset = SA1B_Dataset(root=DATA_DIR.joinpath('SA_1B/images/'))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    teacher = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE).eval()
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    model_type = "vit_t"
    sam_checkpoint = "bin/mobile_sam.pt" if PRETRAINED else None

    model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(DEVICE).train()
    student = SamPredictor(model)

    optimizer = torch.optim.Adam(student.model.parameters(), lr=1e-3)
    distiller = Distiller(teacher, student, processor, dataloader, optimizer, DEVICE)

    distiller.distill()
    torch.save(distiller.student.model.state_dict(), 'bin/distilled_mobile_sam.pt')


if __name__ == "__main__":
    main()