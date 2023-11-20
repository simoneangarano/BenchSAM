from pathlib import Path
import yaml
import json

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
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    json.dump(cfg, open( f"bin/configs/{cfg['MODE']}_{cfg['EXP']}.json",'w'))
    cfg['DATA_DIR'] = Path(cfg['DATA_DIR'])
    cfg['DEVICE'] = torch.device(f"cuda:{cfg['GPU']}" if torch.cuda.is_available() else "cpu")
    cfg['PRETRAINED'] = True if cfg['MODE'] == 'decoder' else False

    dataset = SA1B_Dataset(root=cfg['DATA_DIR'].joinpath('SA_1B/images/'), split=["sa_00000" + str(i) for i in range(cfg['TRAIN_SPLITS'])],
                           features=None, labels=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=cfg['SHUFFLE'], num_workers=cfg['NUM_WORKERS'], pin_memory=False)
    test_dataset = SA1B_Dataset(root=cfg['DATA_DIR'].joinpath('SA_1B/images/'), split=[cfg['SPLIT']], 
                                features=None, labels=True, max=cfg['MAX_TEST'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg['NUM_WORKERS'], pin_memory=False)

    teacher = SamModel.from_pretrained("facebook/sam-vit-huge").to(cfg['DEVICE'])
    teacher.eval()
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    sam_checkpoint = cfg['CKPT'] if cfg['PRETRAINED'] else None
    model = sam_model_registry["vit_t"](checkpoint=sam_checkpoint, size_embedding=cfg['SIZE_EMBEDDING']).to(cfg['DEVICE'])
    model.eval()
    for m in model.image_encoder.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
    student = SamPredictor(model)

    if cfg['MODE'] == 'encoder':
        DISTILLER = EncDistiller
        params = student.model.image_encoder.parameters()
    else:
        DISTILLER = DecDistiller
        params = list(student.model.mask_decoder.parameters()) + list(student.model.prompt_encoder.parameters())

    if cfg['OPTIM'] == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=cfg['LR'], weight_decay=cfg['WD'])
    elif cfg['OPTIM'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=cfg['LR'])

    if cfg['SCHEDULER'] == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['DECAY'])
    elif cfg['SCHEDULER'] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['EPOCHS'], eta_min=1.0e-7)
    elif cfg['SCHEDULER'] == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    distiller = DISTILLER(teacher, student, processor, dataloader, test_dataloader, optimizer, scheduler, 
                          loss_weights=cfg['LOSS_WEIGHTS'], n_prompts=cfg['N_PROMPTS'], random_prompt=cfg['RANDOM_PROMPT'], 
                          edge_filter=cfg['EDGE_FILTER'], size_thr=cfg['SIZE_THR'],
                          profile=cfg['PROFILE'], device=cfg['DEVICE'], debug=cfg['DEBUG'])
    
    if cfg['MODE'] == 'save_features':
        distiller.save_teacher_features(Path('results/teacher_features.pt'))
    else:
        distiller.distill(epochs=cfg['EPOCHS'], acc=cfg['BATCH_SIZE'], use_saved_features=cfg['LOAD_FEATURES'], name=f"{cfg['MODE']}_{cfg['EXP']}")



if __name__ == "__main__":
    main()