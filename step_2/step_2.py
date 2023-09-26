import argparse, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torchvision import transforms, datasets
from datasets import CitySegmentation
from transformers import (SamModel, SamProcessor)

from utils import annToMask

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter('ignore', FutureWarning)



class SinglePointInferenceEngine:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.data_dir = Path(args.data_dir)
        self.output_dir = Path(args.output_dir)
        self.sparse_dir = Path(args.sparse_dir)
        self.prompt_dir = self.output_dir.joinpath(f'{args.dataset}_prompts.pkl')
        
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
            annFile = self.data_dir.joinpath(f'annotations/instances_{self.args.split}2017.json')
            dataset = datasets.CocoDetection(root=self.data_dir.joinpath(f'{self.args.split}2017'), 
                                             annFile=annFile, 
                                             transform=transforms.ToTensor(),
                                             target_transform=None,
                                             transforms=None)
        else:
            raise NotImplementedError

        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, 
                                                      num_workers=self.args.num_workers, pin_memory=self.args.pin_memory, worker_init_fn=None)

    def get_model(self):
        if self.args.sparsity > 0:
            print('Loading sparse model...')
            self.model = SamModel.from_pretrained(self.sparse_dir.joinpath(f'{self.args.sparsity}')).to(self.device).eval()
        else:
            print('Loading dense model...')
            self.model = SamModel.from_pretrained(self.args.model).to(self.device).eval()
        self.processor = SamProcessor.from_pretrained(self.args.processor)
    
    # Inference

    def get_output(self, img, prompt):
        inputs = self.processor(img, input_points=prompt, return_tensors="pt").to(self.device)
        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        inputs.pop("pixel_values", None) # pixel_values are no more needed
        inputs.update({"image_embeddings": image_embeddings})
        with torch.no_grad():
            outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        scores = outputs.iou_scores
        return masks, scores
        
    def get_prompt(self, image, label):
        # Use precomputed prompts if available
        try:
            label[0]['image_id']
        except:
            print('Empty label')
            return None, None

        if self.prompts is not None:
            prompt = self.prompts[self.prompts['name']==label[0]['image_id']][['prompt', 'class']]
            if prompt.values.size > 0:
                return [[prompt.values[0][0]]], prompt.values[0][1] 
            else:
                return None, None
        # Otherwise, generate a new prompt

        m = random.sample(label, 1)[0]

        c = m['category_id']
        m = annToMask(image, m)

        x_v, y_v = np.where(m >= 0)
        r = random.randint(0,len(x_v))
        x, y = x_v[r], y_v[r]
        return [[[y,x]]], c

    def get_masks(self):
        name_list, mask_list, score_list, prompt_list, p_class_list = [], [], [], [], []
        for i, l in self.dataloader:
            
            prompt, p_class = self.get_prompt(i, l)
            if prompt is None:
                continue

            masks, scores = self.get_output(i, prompt)
            
            name_list.append(int(l[0]['image_id'].cpu().detach().numpy()))
            mask_list.append(masks.squeeze()[scores.argmax()].cpu().detach().numpy())
            score_list.append(float(scores.max().cpu().detach().numpy()))
            prompt_list.append(prompt[0][0])
            p_class_list.append(int(p_class))

        if self.prompts is None:
            self.prompts = pd.DataFrame({'name': name_list, 'prompt': prompt_list, 'class': p_class_list})
            self.prompts.to_pickle(self.prompt_dir) 
            print('Prompts saved to file...')

        return name_list, prompt_list, p_class_list, mask_list, score_list



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/simone/SAM/COCO/')
    parser.add_argument('--sparse_dir', type=str, default='/home/simone/SAM/sparsam/bin/')
    parser.add_argument('--output_dir', type=str, default='../results')

    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'cityscapes'])
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--model', type=str, default='facebook/sam-vit-huge')
    parser.add_argument('--processor', type=str, default='facebook/sam-vit-huge')
    parser.add_argument('--sparsity', type=int, default=90)

    parser.add_argument('--save_results', type=bool, default=True)

    args = parser.parse_args()
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu')

    spie = SinglePointInferenceEngine(args, device)
    print('Extracting masks...')
    name, prompt, p_class, mask, score = spie.get_masks()
    if args.save_results:
        print('Saving results...')
        df = pd.DataFrame({'name': name, 'prompt': prompt, 'class': p_class, 'mask': mask, 'score': score})
        df.to_pickle(spie.output_dir.joinpath(f'{args.dataset}_SAM_{args.sparsity}.pkl'))
        print('Results saved!')

if __name__ == '__main__':
    main()