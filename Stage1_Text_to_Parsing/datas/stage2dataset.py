from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import copy
import csv
import cv2
import os
import sys
import clip
import PIL


class TrainDataset(Dataset):
    def __init__(self, path1,path2,path3, path4, image_key="image", caption_key="txt"):
        self.parsing_path  = path1
        self.parsing_files = os.listdir(self.parsing_path)
        self.parsing_files.sort()
        self.parsing_files = self.parsing_files


        self.parsing_wo_cloth_path  = path2
        self.parsing_wo_cloth_files = os.listdir(self.parsing_wo_cloth_path)
        self.parsing_wo_cloth_files.sort()
        self.parsing_wo_cloth_files = self.parsing_wo_cloth_files
        


        self.densepose_path  = path3
        self.densepose_files = os.listdir(self.densepose_path)
        self.densepose_files.sort()
        self.densepose_files = self.densepose_files

        self.text_path  = path4
        with open(path4, 'r') as caption_file:
            self.caption_list = json.load(caption_file)




        self.tform = transforms.ToTensor()
        self.image_key = image_key
        self.caption_key = caption_key
        self.concat_key = 'concat'



    def __len__(self):
        return len(self.parsing_files)

    def __getitem__(self, idx):
        parsing = self.tform(Image.open(self.parsing_path+'/'+self.parsing_files[idx]).convert("RGB"))
        parsing = parsing * 2. - 1.

        parsing_wo_cloth = self.tform(Image.open(self.parsing_wo_cloth_path+'/'+self.parsing_wo_cloth_files[idx]).convert("RGB"))
        parsing_wo_cloth = parsing_wo_cloth * 2. - 1.

        densepose = self.tform(Image.open(self.densepose_path+'/'+self.densepose_files[idx]).convert("RGB"))
        densepose = densepose * 2. - 1.

        caption = self.caption_list[self.parsing_files[idx][:-6]+'.jpg']


        

        
        return {self.image_key: parsing, self.concat_key:[parsing_wo_cloth,densepose], self.caption_key: caption}


class TestDataset(Dataset):
    def __init__(self, path1,path2,path3, path4, image_key="image", caption_key="txt"):
        self.parsing_path  = path1
        self.parsing_files = os.listdir(self.parsing_path)
        self.parsing_files.sort()
        self.parsing_files = self.parsing_files

        self.parsing_wo_cloth_path  = path2
        self.parsing_wo_cloth_files = os.listdir(self.parsing_wo_cloth_path)
        self.parsing_wo_cloth_files.sort()
        self.parsing_wo_cloth_files = self.parsing_wo_cloth_files
        


        self.densepose_path  = path3
        self.densepose_files = os.listdir(self.densepose_path)
        self.densepose_files.sort()
        self.densepose_files = self.densepose_files

        self.text_path  = path4
        with open(path4, 'r') as caption_file:
            self.caption_list = json.load(caption_file)




        self.tform = transforms.ToTensor()
        self.image_key = image_key
        self.caption_key = caption_key
        self.concat_key = 'concat'



    def __len__(self): 
        return len(self.parsing_files)

    def __getitem__(self, idx):
        parsing = self.tform(Image.open(self.parsing_path+'/'+self.parsing_files[idx]).convert("RGB"))
        parsing = parsing * 2. - 1.

        parsing_wo_cloth = self.tform(Image.open(self.parsing_wo_cloth_path+'/'+self.parsing_files[idx]).convert("RGB"))
        parsing_wo_cloth = parsing_wo_cloth * 2. - 1.

        densepose = self.tform(Image.open(self.densepose_path+'/'+self.parsing_files[idx]).convert("RGB"))
        densepose = densepose * 2. - 1.


        caption = self.caption_list[self.parsing_files[idx]]


        

        
        return {self.image_key: parsing, self.concat_key:[parsing_wo_cloth,densepose], self.caption_key: caption, 'file_name':self.parsing_files[idx]}
    
