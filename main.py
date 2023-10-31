import argparse, random, warnings
from pathlib import Path
from tqdm import tqdm
import gc

import numpy as np
import pandas as pd
import cv2

import torch
from datasets import SA1B_Dataset, CitySegmentation, COCOSegmentation
from fastsam import FastSAM, FastSAMPrompt
from transformers import SamModel, SamProcessor
from mobile_sam import sam_model_registry, SamPredictor
from utils import get_mask_limits

warnings.simplefilter('ignore', FutureWarning)
gc.collect()



class SinglePointInferenceEngine:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.model_dir = Path(args.model_dir)
        self.prompt_dir = self.output_dir.joinpath(f'{args.experiment}{args.dataset}_prompts.pkl')
        
        if self.prompt_dir.exists():
            self.prompts = pd.read_pickle(self.prompt_dir) 
            print('Prompts loaded from file...') 
        else:
            self.prompts = None
            print('Random Prompts...') 

        self.get_dataloader()
        self.get_model()


    def get_dataloader(self):
        if self.args.dataset == 'cityscapes':
            dataset = CitySegmentation(root=self.data_dir.joinpath('Cityscapes'), split=self.args.split, crop_size=self.args.crop_size)
        elif self.args.dataset == 'coco':
            dataset = COCOSegmentation(root=self.data_dir.joinpath('coco-2017/'), split=self.args.split, crop_size=self.args.crop_size)
        elif self.args.dataset == 'sa1b':
            dataset = SA1B_Dataset(root=self.data_dir.joinpath('SA_1B/images/'))
        else:
            raise NotImplementedError

        self.classes = dataset.classes
        self.n_classes = dataset.num_class

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, 
                                                      num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, worker_init_fn=None)


    def get_model(self):
        if self.args.model == 'SAM':
            if self.args.sparsity > 0:
                print('Loading sparse model...')
                if self.args.pruning_method == 'l1norm':
                    self.model = SamModel.from_pretrained(self.model_dir.joinpath(f'GUP_L1_{self.args.sparsity}')).to(self.device).eval()
                elif self.args.pruning_method == 'sparsegpt':
                    self.model = SamModel.from_pretrained(self.model_dir.joinpath(f'{self.args.sparsity}')).to(self.device).eval()
            else:
                print('Loading dense model...')
                self.model = SamModel.from_pretrained('facebook/sam-vit-huge').to(self.device).eval()
            self.processor = SamProcessor.from_pretrained('facebook/sam-vit-huge')
        elif self.args.model == 'MobileSAM':
            weights = self.args.weights if self.args.weights is not None else "mobile_sam.pt"
            print(f'Loading MobileSAM model {weights}')
            sam_checkpoint = self.model_dir.joinpath(weights)
            predictor = sam_model_registry['vit_t'](checkpoint=sam_checkpoint).to(self.device).eval()
            self.model = SamPredictor(predictor)
        elif self.args.model == 'FastSAM':
            print('Loading FastSAM model...')
            self.model = FastSAM(self.model_dir.joinpath('FastSAM.pt'))
            self.model.to(self.device)
        else:
            raise NotImplementedError


    def get_output(self, img, prompt):
        if self.args.model == 'SAM':
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
        elif self.args.model == 'MobileSAM':
            img = img[0].detach().cpu().numpy().astype(np.uint8)
            self.model.set_image(img)
            masks, scores, _ = self.model.predict(np.array(prompt[0]), np.array([1]))
            mask = masks.squeeze()[scores.argmax()]
            score = float(scores.max())
        elif self.args.model == 'FastSAM':
            img = img[0].detach().cpu().numpy().astype(np.uint8)
            everything_results = self.model(img, device=self.device, verbose=False) # DEPRECATED ARGS
                                            # retina_masks=self.args.retina, imgsz=self.args.imsize, 
                                            # conf=self.args.conf_thr, iou=self.args.iou_thr)
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
            return [[prompt.values[0][0]]], prompt.values[0][1] 
            
        # Otherwise, generate a new prompt
        if self.args.edge_filter:
            e = cv2.Canny(image=label.numpy().astype(np.uint8), threshold1=10, threshold2=50)
            e = cv2.dilate(e, np.ones((self.args.edge_width, self.args.edge_width), np.uint8), iterations = 1).astype(bool)
            label[e] = 0

        C = np.unique(label)[1:]
        if len(C) == 0:
            c = 0
        else:
            c = np.random.choice(C)
        x_v, y_v = np.where(label == c)
        r = random.randint(0, len(x_v) - 1)
        x, y = x_v[r], y_v[r]

        return [[[y,x]]], c # inverted to compensate different indexing


    def get_pred_classes(self, inst, label):
        im = np.logical_not(inst).astype(np.uint8)
        im[im==1] = self.n_classes
        m = label + im
        h, _ = np.histogram(m, bins=256, range=(0,255))
        clean_h = h[:self.n_classes]
        mask_tot = np.sum(clean_h)
        classes = np.where(clean_h > self.args.class_thr * mask_tot)[0]
        return list(classes)


    def get_masks(self):
        name_list, mask_list, score_list, prompt_list = [], [], [], []
        p_class_list, s_class_list, origin_list, shape_list = [], [], [], []

        origin = [0,0]
        for _, (i, l, n) in enumerate(tqdm(self.dataloader)):
            
            prompt, p_class = self.get_prompt(n[0], l[0])
            mask, score = self.get_output(i, prompt)
            if self.args.crop_mask:
                origin, _, mask = get_mask_limits([mask])

            name_list.append(n[0])
            shape_list.append(i.shape[2:])
            mask_list.append(mask)
            origin_list.append(origin[::-1])
            score_list.append(score)
            prompt_list.append(prompt[0][0])
            if self.args.dataset == 'sa1b':
                p_class_list.append(None)
                s_class_list.append(None)
            else:
                p_class_list.append(int(p_class))
                s_class_list.append(list(self.get_pred_classes(mask, l)))

        if self.prompts is None:
            self.prompts = pd.DataFrame({'name': name_list, 'prompt': prompt_list, 'class': p_class_list,
                                         'image_shape': shape_list})
            self.prompts.to_pickle(self.prompt_dir) 
            print(f'Prompts saved to file {self.prompt_dir}')

        return name_list, prompt_list, p_class_list, s_class_list, mask_list, origin_list, score_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../Datasets/')
    parser.add_argument('--model_dir', type=str, default='bin/')
    parser.add_argument('--output_dir', type=str, default='results/')

    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'cityscapes', 'sa1b'])
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--model', type=str, default='SAM', choices=['SAM', 'MobileSAM', 'FastSAM'])
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--sparsity', type=int, default=0)
    parser.add_argument('--pruning_method', type=str, default='l1norm', choices=['l1norm', 'sparsegpt'])

    # Deprecated
    # parser.add_argument('--imsize', type=int, default=1024)
    # parser.add_argument('--retina', type=bool, default=True)
    # parser.add_argument('--conf_thr', type=float, default=0.25)
    # parser.add_argument('--iou_thr', type=float, default=0.0)
    # parser.add_argument('--center_prompt', type=bool, default=False) # if True, the prompt is the centroid of the instance mask

    parser.add_argument('--class_thr', type=float, default=0.05) # ignores classes with less than 5% of the instance mask
    parser.add_argument('--edge_filter', type=bool, default=False) # removes edges from the instance mask before computing the prompt
    parser.add_argument('--edge_width', type=int, default=5) # width of the border to remove

    parser.add_argument('--crop_mask', type=bool, default=False) # if True, the mask is cropped to the instance limits
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--experiment', type=str, default='')

    args = parser.parse_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')    

    spie = SinglePointInferenceEngine(args, device)
    print('Extracting masks...')
    name, prompt, p_class, s_class, mask, origin, score = spie.get_masks()

    if args.save_results:
        print('Saving results...')
        df = pd.DataFrame({'name': name, 'prompt': prompt, 'class': p_class, 's_class': s_class, 
                           'mask': mask, 'mask_origin': origin, 'score': score})
        p = '_gup' if args.pruning_method == 'l1norm' else ''
        out_file = spie.output_dir.joinpath(f'{args.experiment}{args.dataset}_{args.model}_{args.sparsity}{p}.pkl')
        df.to_pickle(out_file)
        print(f'Results saved! {out_file}')

        del df, spie
        gc.collect()



if __name__ == '__main__':
    main()
