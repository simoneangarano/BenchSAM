from pathlib import Path

import torch
from transformers import (SamModel, SamProcessor)
from utils.mobile_sam import sam_model_registry

import sys
sys.path.append('..')
from utils.mobile_sam.predictor import SamPredictor
from utils.datasets import SA1B_Dataset
from utils.utils import *
from utils.distill_utils import *



def main():

    DATA_DIR = Path('../Datasets/')
    SPLIT = 'sa_000020'
    GPU = 4
    DEVICE = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    TRAIN_SPLITS = 1
    MAX_TEST = 2500
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    SHUFFLE = True
    LOAD_FEATURES = True
    # FEATURES = 'results/teacher_features.pt' if LOAD_FEATURES else None

    EPOCHS = 20
    LR = 3e-4
    OPTIM = 'adamw'
    SCHEDULER = 'cos'
    DECAY = 1.0
    WD = 1e-5
    LOSS_WEIGHTS = [0,0,1,1,0,0,0] # FW, DW, BW, IW, MW, SW, KW
    SIZE_EMBEDDING = 'none' # sparse, dense
    
    MODE = 'decoder' # encoder, decoder, save_features
    PRETRAINED = True if MODE == 'decoder' else False
    EXP = 'distill_bce'
    PROFILE = False

    dataset = SA1B_Dataset(root=DATA_DIR.joinpath('SA_1B/images/'), split=["sa_00000" + str(i) for i in range(TRAIN_SPLITS)],
                           features=None, labels=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=False)
    test_dataset = SA1B_Dataset(root=DATA_DIR.joinpath('SA_1B/images/'), split=[SPLIT], features=None, labels=True, max=MAX_TEST)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    teacher = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
    teacher.eval()
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    model_type = "vit_t"
    sam_checkpoint = "bin/mobile_sam.pt" if PRETRAINED else None
    model = sam_model_registry[model_type](checkpoint=sam_checkpoint, size_embedding=SIZE_EMBEDDING).to(DEVICE)
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
        params = list(student.model.mask_decoder.parameters()) + list(student.model.prompt_encoder.parameters())

    if OPTIM == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    elif OPTIM == 'adam':
        optimizer = torch.optim.Adam(params, lr=LR)

    if SCHEDULER == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY)
    elif SCHEDULER == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1.0e-7)
    elif SCHEDULER == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    distiller = DISTILLER(teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, loss_weights=LOSS_WEIGHTS, 
                          profile=PROFILE, device=DEVICE)

    if MODE == 'save_features':
        distiller.save_teacher_features(Path('results/teacher_features.pt'))
    else:
        distiller.distill(epochs=EPOCHS, accumulate=BATCH_SIZE, use_saved_features=LOAD_FEATURES, name=f'{MODE}_{EXP}')



if __name__ == "__main__":
    main()