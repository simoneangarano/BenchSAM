import os
import argparse, random, warnings
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from datasets import CitySegmentation, COCOSegmentation

from fastsam import FastSAM, FastSAMPrompt
from utils import suppress_stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter('ignore', FutureWarning)



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
        print('Dataloader initialized...')
        self.get_model()
        print('Model initialized...')

    # Setup

    def get_dataloader(self):
        if self.args.dataset == 'cityscapes':
            dataset = CitySegmentation(root=self.args.data_dir, split=self.args.split, crop_size=self.args.crop_size)
        elif self.args.dataset == 'coco':
            dataset = COCOSegmentation(root=self.args.data_dir, split=self.args.split, crop_size=self.args.crop_size)
        else:
            raise NotImplementedError

        self.classes = dataset.classes
        self.n_classes = dataset.num_class

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, 
                                                      num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, worker_init_fn=None)

    def get_model(self):
        self.model = FastSAM(self.model_dir.joinpath('FastSAM.pt'))
    
    # Inference

    def get_output(self, img, prompt):

        with suppress_stdout():
            everything_results = self.model(img, device=self.device)
            prompt_process = FastSAMPrompt(img, everything_results, device=self.device)
            mask, score = prompt_process.point_prompt(points=prompt, pointlabel=[1])
        return mask, score
        
    def get_prompt(self, name, label):
        # Use precomputed prompts if available
        if self.prompts is not None:
            prompt = self.prompts[self.prompts['name']==int(name[0])][['prompt', 'class']]
            return [prompt.values[0][0]], prompt.values[0][1] 
            
        # Otherwise, generate a new prompt
        C = np.unique(label[0])[1:]
        c = np.random.choice(C)
        if self.args.center_prompt:
            x, y = (torch.sum(torch.argwhere(label[0]==c),0)/torch.sum(label[0]==c)).detach().cpu().numpy()
            x, y = int(x), int(y)
        else:
            x_v, y_v = np.where(label[0] == c)
            r = random.randint(0, len(x_v) - 1)
            x, y = x_v[r], y_v[r]

        return [[y,x]], c # inverted to compensate different indexing

    def get_pred_classes(self, inst, label):
        im = torch.logical_not(inst).to(torch.uint8)
        im[im==1] = self.n_classes
        m = im + label
        h, _ = np.histogram(m, bins=256, range=(0,255))
        clean_h = h[:self.n_classes]
        mask_tot = np.sum(clean_h)
        classes = np.where(clean_h > self.args.class_thr * mask_tot)[0]
        return list(classes)

    def get_masks(self):
        name_list, mask_list, score_list, prompt_list, p_class_list, s_class_list = [], [], [], [], [], []
        for j, (i, l, n) in enumerate(tqdm(self.dataloader)):
            
            prompt, p_class = self.get_prompt(n, l)
            masks, scores = self.get_output(i[0].detach().cpu().numpy().astype(np.uint8), prompt)
            
            m = torch.nn.functional.upsample(masks[None,None].to(torch.uint8),[i.shape[1], i.shape[2]], mode='nearest')[0,0].bool()

            name_list.append(int(n[0]))
            mask_list.append(m.cpu().detach().numpy())
            score_list.append(float(scores))
            prompt_list.append(prompt[0])
            p_class_list.append(int(p_class))
            s_class_list.append(list(self.get_pred_classes(m, l)))

        if self.prompts is None:
            self.prompts = pd.DataFrame({'name': name_list, 'prompt': prompt_list, 'class': p_class_list})
            self.prompts.to_pickle(self.prompt_dir) 
            print('Prompts saved to file...')

        return name_list, prompt_list, p_class_list, s_class_list, mask_list, score_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../Datasets/coco-2017/')
    parser.add_argument('--model_dir', type=str, default='../bin/')
    parser.add_argument('--output_dir', type=str, default='../results')

    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'cityscapes'])
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--model', type=str, default='facebook/sam-vit-huge')
    parser.add_argument('--processor', type=str, default='facebook/sam-vit-huge')
    parser.add_argument('--sparsity', type=int, default=0)

    parser.add_argument('--center_prompt', type=bool, default=False)
    parser.add_argument('--class_thr', type=float, default=0.05)

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
    name, prompt, p_class, s_class, mask, score = spie.get_masks()

    if args.save_results:
        print('Saving results...')
        df = pd.DataFrame({'name': name, 'prompt': prompt, 'class': p_class, 's_class': s_class, 'mask': mask, 'score': score})
        df.to_pickle(spie.output_dir.joinpath(f'{args.experiment}{args.dataset}_SAM_{args.sparsity}.pkl'))
        print('Results saved!')

if __name__ == '__main__':
    main()