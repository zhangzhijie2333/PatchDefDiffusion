import os 
import torch 
import json 
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import numpy as np


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def exist_in(short_str, list_of_string):
    for string in list_of_string:
        if short_str in string:
            return True 
    return False 


def clean_files(image_files, normal_files):
    """
    Not sure why some images do not have normal map annotations, thus delete these images from list. 

    The implementation here is inefficient .....  
    """
    new_image_files = []

    for image_file in image_files:
        image_file_basename = os.path.basename(image_file).split('.')[0]
        if exist_in(image_file_basename,normal_files):
            new_image_files.append(image_file)
    image_files = new_image_files


    # a sanity check 
    for image_file, normal_file in zip(image_files, normal_files):
        image_file_basename = os.path.basename(image_file).split('.')[0]
        normal_file_basename = os.path.basename(normal_file).split('.')[0]
        assert image_file_basename == normal_file_basename[:-7] 
    
    return image_files, normal_files





class GrayGenerationDataset():
    def __init__(self, hint_rootdir, gray_rootdir, caption_path=None, image_size=128, random_flip=False, prob_use_caption=1.0):
        self.hint_rootdir = hint_rootdir
        self.gray_rootdir = gray_rootdir
        self.caption_path = caption_path
        self.image_size = image_size
        self.random_flip = random_flip
        self.prob_use_caption = prob_use_caption  # 新增参数
        
        gray_files = recursively_read(gray_rootdir, must_contain="", exts=['bmp'])
        gray_files.sort()
        hint_files = recursively_read(hint_rootdir, must_contain="", exts=['png'])
        hint_files.sort()

        self.gray_files = gray_files
        self.hint_files = hint_files

        # 加载 caption
        if caption_path:
            with open(caption_path, 'r') as f:
                self.image_filename_to_caption_mapping = json.load(f)
        else:
            self.image_filename_to_caption_mapping = []

        # 获取最小长度
        min_len = min(len(self.gray_files), len(self.hint_files), len(self.image_filename_to_caption_mapping))
        
        # 裁剪数据对齐
        self.gray_files = self.gray_files[:min_len]
        self.hint_files = self.hint_files[:min_len]

        if isinstance(self.image_filename_to_caption_mapping, dict):
            self.image_filename_to_caption_mapping = {
                k: self.image_filename_to_caption_mapping[k] 
                for k in list(self.image_filename_to_caption_mapping.keys())[:min_len]
            }
        elif isinstance(self.image_filename_to_caption_mapping, list):
            self.image_filename_to_caption_mapping = self.image_filename_to_caption_mapping[:min_len]
        else:
            raise TypeError(f"Unexpected type for image_filename_to_caption_mapping: {type(self.image_filename_to_caption_mapping)}")
        
        print(f"Adjusted dataset size to {min_len} to avoid mismatched lengths.")

        self.pil_to_tensor = transforms.PILToTensor()

    def __getitem__(self, index):
        hint_path = self.hint_files[index]
        gray_path = self.gray_files[index]

        hint = Image.open(hint_path).convert("RGB")
        gray = Image.open(gray_path).convert("L")

        assert hint.size == gray.size, f"{hint_path} 和 {gray_path} 尺寸不一致"

        # 中心裁剪 + 缩放
        crop_size = min(hint.size)
        hint = TF.center_crop(hint, crop_size)
        gray = TF.center_crop(gray, crop_size)

        hint = hint.resize((self.image_size, self.image_size))
        gray = gray.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.random_flip and random.random() < 0.5:
            hint = ImageOps.mirror(hint)
            gray = ImageOps.mirror(gray)

        # 标准化 hint 图像为 [-1, 1]
        hint_tensor = (self.pil_to_tensor(hint).float() / 255 - 0.5) / 0.5
        gray_tensor = self.pil_to_tensor(gray).float() / 255  # [0, 1]

        out = {
            "id": index,
            "image_hint": hint_tensor,
            "image": gray_tensor,
            "mask": torch.tensor(1.0)
        }

        # 仅当 self.captions 存在且符合概率时使用 caption
        if random.uniform(0, 1) < self.prob_use_caption:
            # 根据文件名中的关键字设置 caption
            if 'hole' in os.path.basename(hint_path):
                out["caption"] = 'hole'
        else:
            out["caption"] = ""

        return out

    def __len__(self):
        return len(self.hint_files)
        
    # 添加 total_images() 方法
    def total_images(self):
        return len(self)
