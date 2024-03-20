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
        self.image_path  = path1
        self.image_files = os.listdir(self.image_path)
        self.image_files.sort()
        self.image_files = self.image_files


        self.image_wo_cloth_path  = path2
        self.image_wo_cloth_files = os.listdir(self.image_wo_cloth_path)
        self.image_wo_cloth_files.sort()
        self.image_wo_cloth_files = self.image_wo_cloth_files
        


        self.parsing_path  = path3
        self.parsing_files = os.listdir(self.parsing_path)
        self.parsing_files.sort()
        self.parsing_files = self.parsing_files


        self.cloth_patch_path  = path4
        self.cloth_patch_files = os.listdir(self.cloth_patch_path)
        self.cloth_patch_files.sort()
        self.cloth_patch_files = self.cloth_patch_files

        

        self.tform = transforms.ToTensor()
        self.image_key = image_key
        self.caption_key = caption_key
        self.concat_key = 'concat'



    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.tform(Image.open(self.image_path+'/'+self.image_files[idx]).convert("RGB"))
        image = image * 2. - 1.

        image_wo_cloth = self.tform(Image.open(self.image_wo_cloth_path+'/'+self.image_wo_cloth_files[idx]).convert("RGB"))
        image_wo_cloth = image_wo_cloth * 2. - 1.

        parsing = self.tform(Image.open(self.parsing_path+'/'+self.parsing_files[idx]).convert("RGB"))
        parsing = parsing * 2. - 1.


        cloth = self.tform(Image.open(self.cloth_patch_path+'/'+self.cloth_patch_files[idx]).convert("RGB"))
        cloth = cloth * 2. - 1.



        
        return {self.image_key: image, self.concat_key:[image_wo_cloth,parsing], self.caption_key: cloth}


class TestDataset(Dataset):
    def __init__(self, path1,path2,path3, path4, path5, image_key="image", caption_key="txt"):
        self.image_path  = path1
        self.image_files = os.listdir(self.image_path)
        self.image_files.sort()
        self.image_files = self.image_files


        self.image_wo_cloth_path  = path2
        self.image_wo_cloth_files = os.listdir(self.image_wo_cloth_path)
        self.image_wo_cloth_files.sort()
        self.image_wo_cloth_files = self.image_wo_cloth_files
        


        self.parsing_path  = path3
        self.parsing_files = os.listdir(self.parsing_path)
        self.parsing_files.sort()
        self.parsing_files = self.parsing_files


        self.cloth_patch_path  = path4
        self.cloth_patch_files = os.listdir(self.cloth_patch_path)
        self.cloth_patch_files.sort()
        self.cloth_patch_files = self.cloth_patch_files

        self.parsing_original_path  = path5
        self.parsing_original_files = os.listdir(self.parsing_original_path)
        self.parsing_original_files.sort()
        self.parsing_original_files = self.parsing_original_files




        

        self.tform = transforms.ToTensor()
        self.image_key = image_key
        self.caption_key = caption_key
        self.concat_key = 'concat'



    def __len__(self):
        return len(self.parsing_files)-1
        # return 50

    def __getitem__(self, idx):
        image = self.tform(Image.open(self.image_path+'/'+self.parsing_files[idx]).convert("RGB").resize((256,512)))
        image = image * 2. - 1.

        image_wo_cloth = self.tform(Image.open(self.image_wo_cloth_path+'/'+self.parsing_files[idx]).convert("RGB").resize((256,512)))
        image_wo_cloth = image_wo_cloth * 2. - 1.

        parsing = self.tform(Image.open(self.parsing_path+'/'+self.parsing_files[idx]).convert("RGB").resize((256,512)))
        parsing = parsing * 2. - 1.
        

        parsing_origin = np.array(Image.open(self.parsing_original_path+'/'+self.parsing_files[idx]).resize((256,512)))


        cloth = self.tform(Image.open(self.cloth_patch_path+'/'+self.parsing_files[idx+1]).convert("RGB").resize((256,256)))

        cloth = cloth * 2. - 1.
        file_name = self.parsing_files[idx]

        parsing_image = np.array(Image.open(self.parsing_path+'/'+self.parsing_files[idx]).resize((256,512)))
        
        return {self.image_key: image, self.concat_key:[image_wo_cloth,parsing], self.caption_key: cloth,'original_img':parsing_origin,'file_name':file_name,'parsing_image':parsing_image}