import random, warnings
from pathlib import Path
from tqdm import tqdm
import gc, yaml

import numpy as np
import pandas as pd
import cv2

import torch
from pycocotools import mask as maskUtils
from utils.datasets import SA1B_Dataset, CitySegmentation, COCOSegmentation
from utils.fastsam import FastSAM, FastSAMPrompt
from transformers import SamModel, SamProcessor
from utils.mobile_sam import sam_model_registry, SamPredictor
from utils.utils import get_mask_limits

warnings.simplefilter('ignore', FutureWarning)
gc.collect()



class SinglePointInferenceEngine:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.data_dir = Path(cfg['DATA_DIR'])
        self.output_dir = Path(cfg['OUTPUT_DIR'])
        self.model_dir = Path(cfg['MODEL_DIR'])
        self.prompt_dir = self.output_dir.joinpath(f"{cfg['EXP']}{cfg['DATASET']}_prompts.pkl")
        if self.cfg['SIZE_EMBED'] == 'sparse':
            self.targets = pd.read_pickle(f"results/{cfg['EXP']}{cfg['DATASET']}_SAM_0.pkl")
        if self.prompt_dir.exists():
            self.prompts = pd.read_pickle(self.prompt_dir) 
            print('Prompts loaded from file...') 
        else:
            self.prompts = None
            print('Random Prompts...') 

        self.get_dataloader()
        self.get_model()

    def get_dataloader(self):
        if self.cfg['DATASET'] == 'cityscapes':
            dataset = CitySegmentation(root=self.data_dir.joinpath('Cityscapes'), split=self.cfg['SPLIT'], crop_size=self.cfg['CROP_SIZE'])
        elif self.cfg['DATASET'] == 'coco':
            dataset = COCOSegmentation(root=self.data_dir.joinpath('coco-2017/'), split=self.cfg['SPLIT'], crop_size=self.cfg['CROP_SIZE'])
        elif self.cfg['DATASET'] == 'sa1b':
            dataset = SA1B_Dataset(root=self.data_dir.joinpath('SA_1B/images/'), split=["sa_000020"],
                                   features=None, labels=True)
        else:
            raise NotImplementedError

        self.classes = dataset.classes
        self.n_classes = dataset.num_class

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg['BATCH_SIZE'], shuffle=False, 
                                                      num_workers=self.cfg['NUM_WORKERS'], pin_memory=self.cfg['PIN_MEMORY'], worker_init_fn=None)

    def get_model(self):
        if self.cfg['MODEL'] == 'SAM':
            if self.cfg['SPARSITY'] > 0:
                print('Loading sparse model...')
                if self.cfg['PRUNING_METHOD'] == 'l1norm':
                    self.model = SamModel.from_pretrained(self.model_dir.joinpath(f"GUP_L1_{self.cfg['SPARSITY']}")).to(self.device).eval()
                elif self.cfg['PRUNING_METHOD'] == 'sparsegpt':
                    self.model = SamModel.from_pretrained(self.model_dir.joinpath(f"{self.cfg['SPARSITY']}")).to(self.device).eval()
            else:
                print('Loading dense model...')
                self.model = SamModel.from_pretrained('facebook/sam-vit-huge').to(self.device).eval()
            self.processor = SamProcessor.from_pretrained('facebook/sam-vit-huge')
        elif self.cfg['MODEL'] == 'MobileSAM':
            weights = self.cfg['CKPT'] if self.cfg['CKPT'] is not None else "mobile_sam.pt"
            print(f"Loading MobileSAM model {weights}")
            sam_checkpoint = self.model_dir.joinpath(weights)
            predictor = sam_model_registry['vit_t'](checkpoint=sam_checkpoint, add_prompt=self.cfg['SIZE_EMBED']).to(self.device).eval()
            self.model = SamPredictor(predictor)
        elif self.cfg['MODEL'] == 'FastSAM':
            print('Loading FastSAM model...')
            self.model = FastSAM(self.model_dir.joinpath('FastSAM.pt'))
            self.model.to(self.device)
        else:
            raise NotImplementedError


    def get_output(self, img, prompt, label, name):
        if self.cfg['MODEL'] == 'SAM':
            inputs = self.processor(img, input_points=prompt, return_tensors="pt").to(self.device)
            image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
            inputs.pop("pixel_values", None) # pixel_values are no more needed
            inputs.update({"image_embeddings": image_embeddings})
            with torch.no_grad():
                outputs = self.model(**inputs)
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
            scores = outputs.iou_scores
            mask = masks.squeeze()[scores.argmax()].cpu().detach().numpy()
            score = float(scores.max().cpu().detach().numpy())
        elif self.cfg['MODEL'] == 'MobileSAM':
            img = img[0].detach().cpu().numpy().astype(np.uint8)
            self.model.set_image(img)
            size = self.targets[self.targets['name']==name]['mask_size'].values[0] if self.cfg['SIZE_EMBED'] == 'sparse' else 0
            masks, scores, lowres = self.model.predict(np.array(prompt[0]), np.array([1]), size=size, return_logits=self.cfg['SIGMOID'])
            if self.cfg['REFEED']:
                lowres = lowres[scores.argmax()][None]
                masks, scores, lowres = self.model.predict(np.array(prompt[0]), np.array([1]), size=size,
                                                           mask_input=lowres, return_logits=self.cfg['SIGMOID'])
            mask = masks.squeeze()[scores.argmax()]
            if self.cfg['SIGMOID']:
                mask = torch.sigmoid(mask).detach().cpu().numpy() > 0.5
            score = float(scores.max())
        elif self.cfg['MODEL'] == 'FastSAM':
            img = img[0].detach().cpu().numpy().astype(np.uint8)
            everything_results = self.model(img, device=self.device, verbose=False) # DEPRECATED ARGS
                                            # retina_masks=self.cfg['RETINA'], imgsz=self.cfg['IMSIZE'], 
                                            # conf=self.cfg['CONF_THR'], iou=self.cfg['IOU_THR'])
            prompt_process = FastSAMPrompt(img, everything_results, device=self.device)
            mask, score = prompt_process.point_prompt(points=prompt[0], pointlabel=[1])
            mask = torch.nn.functional.interpolate(mask[None,None].to(torch.uint8),[img.shape[0], img.shape[1]], 
                                                   mode='nearest')[0,0].bool().cpu().detach().numpy()
            score = float(score)
        else:
            raise NotImplementedError
        return mask, score
        

    def get_prompt(self, name, label):
        # Use precomputed prompts if available
        if self.prompts is not None:
            prompt = self.prompts[self.prompts['name']==name][['prompt', 'class']]
            return [[prompt.values[0][0]]], label[prompt.values[0][0][1],prompt.values[0][0][0]]
            
        # Otherwise, generate a new prompt
        if self.cfg['EDGE_FILTER']:
            e = cv2.Canny(image=label.numpy().astype(np.uint8), threshold1=10, threshold2=50)
            e = cv2.dilate(e, np.ones((self.cfg['EDGE_WIDTH'], self.cfg['EDGE_WIDTH']), np.uint8), iterations=1).astype(bool)
            label[e] = 0

        C, counts = np.unique(label.cpu(), return_counts=True)
        counts = (counts / counts.sum())[1:]
        Cf = C[1:][(counts > self.size_thr_low) * (counts < self.size_thr_high)]
        if len(Cf) == 0:
            Cf = C[1:][counts > self.size_thr_low]
        c = np.random.choice(Cf)

        x_v, y_v = np.where(label == c)
        if self.cfg['RANDOM_PROMPT']:
            r = random.randint(0, len(x_v) - 1)
            x, y = x_v[r], y_v[r]
        else: # central prompt
            x, y = x_v.mean(), y_v.mean()
            x, y = int(x), int(y)
        return [[[y,x]]], c # inverted to compensate different indexing


    def get_pred_classes(self, inst, label):
        im = np.logical_not(inst).astype(np.uint8)
        im[im==1] = self.n_classes
        m = label + im
        h, _ = np.histogram(m, bins=256, range=(0,255))
        clean_h = h[:self.n_classes]
        mask_tot = np.sum(clean_h)
        classes = np.where(clean_h > self.cfg['CLASS_THR'] * mask_tot)[0]
        return list(classes)


    def get_masks(self):
        name_list, mask_list, score_list, prompt_list = [], [], [], []
        p_class_list, s_class_list, origin_list, shape_list = [], [], [], []

        origin = [0,0]
        for _, sample in enumerate(tqdm(self.dataloader)):
            if self.cfg['DATASET'] == 'sa1b':
                (i, l, n, _) = sample
            else:
                (i, l, n) = sample
            prompt, p_class = self.get_prompt(n[0], l[0])
            mask, score = self.get_output(i, prompt, l[0]==p_class, n[0])
            if self.cfg['CROP_MASK']:
                origin, _, mask = get_mask_limits([mask])

            name_list.append(n[0])
            shape_list.append(i.shape[1:3])
            origin_list.append(origin[::-1])
            score_list.append(score)
            prompt_list.append(prompt[0][0])
            if self.cfg['DATASET'] == 'sa1b':
                p_class_list.append(None)
                s_class_list.append(None)
            else:
                p_class_list.append(int(p_class))
                s_class_list.append(list(self.get_pred_classes(mask, l)))
            if self.cfg['RLE_ENCODING']:
                mask = maskUtils.encode(np.asfortranarray(mask))
            mask_list.append(mask)

        if self.prompts is None:
            self.prompts = pd.DataFrame({'name': name_list, 'prompt': prompt_list, 'class': p_class_list,
                                         'shape': shape_list})
            self.prompts.to_pickle(self.prompt_dir) 
            print(f"Prompts saved to file {self.prompt_dir}")

        return name_list, prompt_list, p_class_list, s_class_list, mask_list, origin_list, score_list



def main():
    with open('config_inference.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    torch.manual_seed(cfg['SEED'])
    np.random.seed(cfg['SEED'])
    random.seed(cfg['SEED'])
    device = torch.device(f"cuda:{cfg['GPU']}" if torch.cuda.is_available() else 'cpu')    

    spie = SinglePointInferenceEngine(cfg, device)
    print('Extracting masks...')
    name, prompt, p_class, s_class, mask, origin, score = spie.get_masks()

    if cfg['SAVE_RESULTS']:
        print('Saving results...')
        df = pd.DataFrame({'name': name, 'prompt': prompt, 'class': p_class, 's_class': s_class, 
                           'mask': mask, 'mask_origin': origin, 'score': score})
        p = '_gup' if cfg['PRUNING_METHOD'] == 'l1norm' else ''
        out_file = spie.output_dir.joinpath(f"{cfg['EXP']}{cfg['DATASET']}_{cfg['MODEL']}{cfg['SUFFIX']}_{cfg['SPARSITY']}{p}.pkl")
        df.to_pickle(out_file)
        print(f"Results saved! {out_file}")

        del df, spie
        gc.collect()



if __name__ == '__main__':
    main()
