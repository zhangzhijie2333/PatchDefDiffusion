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
            if file.lower().split('.')[-1] in [e.lower() for e in exts]  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def exist_in(short_str, list_of_string):
    for string in list_of_string:
        if short_str in string:
            return True 
    return False 


def clean_files(image_files: object, normal_files: object) -> object:
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

class EdgeDataset():
    def __init__(self, image_rootdir, edge_rootdir, caption_path, prob_use_caption=1, image_size=512, random_flip=False):
        self.image_rootdir = image_rootdir
        self.edge_rootdir = edge_rootdir
        self.caption_path = caption_path
        self.prob_use_caption = prob_use_caption 
        self.image_size = image_size
        self.random_flip = random_flip

        # Image and normal files 
        image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=['bmp', 'jpg', 'jpeg', 'png'])
        image_files.sort()
        edge_files = recursively_read(rootdir=edge_rootdir, must_contain="", exts=['bmp', 'jpg', 'jpeg', 'png'])
        edge_files.sort()

        self.image_files = image_files
        self.edge_files = edge_files

        with open(caption_path, 'r') as f:
            self.image_filename_to_caption_mapping = json.load(f)

        min_len = min(len(self.image_files), len(self.edge_files), len(self.image_filename_to_caption_mapping))

        self.image_files = self.image_files[:min_len]
        self.edge_files = self.edge_files[:min_len]

        if isinstance(self.image_filename_to_caption_mapping, dict):
            self.image_filename_to_caption_mapping = {k: self.image_filename_to_caption_mapping[k] for k in list(self.image_filename_to_caption_mapping.keys())[:min_len]}
        elif isinstance(self.image_filename_to_caption_mapping, list):
            self.image_filename_to_caption_mapping = self.image_filename_to_caption_mapping[:min_len]
        else:
            raise TypeError(f"Unexpected type for image_filename_to_caption_mapping: {type(self.image_filename_to_caption_mapping)}")

        print(f"Adjusted dataset size to {min_len} to avoid mismatched lengths.")

        self.pil_to_tensor = transforms.PILToTensor()

    def total_images(self):
        return len(self)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        out = {}

        out['id'] = index
        image = Image.open(image_path).convert("RGB")
        edge = Image.open(self.edge_files[index]).convert("L")

        assert image.size == edge.size

        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize((self.image_size, self.image_size))

        edge = TF.center_crop(edge, crop_size)
        edge = edge.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.random_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)
            edge = ImageOps.mirror(edge)

        edge = self.pil_to_tensor(edge)[0, :, :]

        # Set different label values for five types
        basename = os.path.basename(image_path)
        if 'missing' in basename:
            edge = edge / 255 * 1
        elif 'short' in basename:
            edge = edge / 255 * 2
        elif 'open' in basename:
            edge = edge / 255 * 3
        elif 'mouse' in basename:
            edge = edge / 255 * 4
        elif 'spur_' in basename :
            edge = edge / 255 * 5
        elif 'spurious_' in basename:
            edge = edge / 255 * 6
        elif 'normal' in basename:
            edge = edge / 255 * 7

        input_label = torch.zeros(152, self.image_size, self.image_size)
        edge = input_label.scatter_(0, edge.long().unsqueeze(0), 1.0)

        out['image'] = (self.pil_to_tensor(image).float() / 255 - 0.5) / 0.5
        out['edge'] = edge
        out['mask'] = torch.tensor(1.0)

        # ----- Caption logic -----
        if random.uniform(0, 1) < self.prob_use_caption:
            if 'missing' in basename:
                out["caption"] = 'missing'
            elif 'short' in basename:
                out["caption"] = 'short'
            elif 'open' in basename:
                out["caption"] = 'open'
            elif 'mouse' in basename:
                out["caption"] = 'mouse'
            elif 'spur_' in basename :
                out["caption"] = 'spur'
            elif 'spurious_' in basename:
                out["caption"] = 'spurious'
            elif 'normal' in basename:
                out["caption"] = 'normal'
        else:
            out["caption"] = ""

        return out

    def __len__(self):
        return len(self.image_files)
