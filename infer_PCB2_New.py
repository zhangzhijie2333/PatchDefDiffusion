import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim_reconstruction import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from images_process.edge_extraction import smooth_pcb_edge_extraction
import random
from PIL import Image
import cv2
import torch.utils.checkpoint
from tqdm import tqdm

device = torch.device("cuda")
torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args)

def list_files(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale

def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas

def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path,weights_only=False)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config

def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)

def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature

def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask

def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))
    return labelmap_rgb

@torch.no_grad()
def prepare_batch_edge(meta, batch=1):

    pil_to_tensor = transforms.PILToTensor()
    edge_np = smooth_pcb_edge_extraction(meta['image'])
    if edge_np is None:
        raise ValueError(f"Edge map extraction failed, original image path: {meta['image']}")
    
    edge = Image.fromarray(edge_np).convert("L")
    edge = TF.center_crop(edge, min(edge.size))
    edge = edge.resize((256, 256), Image.NEAREST)

    try:
        edge_color = colorEncode(np.array(edge), loadmat('color150.mat')['colors'])
        Image.fromarray(edge_color).save("edge_vis.png")
    except:
        pass 
    edge = pil_to_tensor(edge)[0,:,:] 
    if 'missing' in meta["prompt"]:
        edge = edge / 255
    elif 'short' in meta["prompt"]:
        edge = edge / 255 * 2
    elif 'open' in meta["prompt"]:
        edge = edge / 255 * 3
    elif 'mouse' in meta["prompt"]:
        edge = edge / 255 * 4
    elif 'spur' in meta["prompt"]:
        edge = edge / 255 * 5
    elif 'spurious' in meta["prompt"]:
        edge = edge / 255 * 6
    elif 'normal' in meta["prompt"]:
        edge = edge / 255 * 7

    input_label = torch.zeros(152, 256, 256)
    edge = input_label.scatter_(0, edge.long().unsqueeze(0), 1.0)

    out = {
        "edge" : edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 

def disable_checkpoint(model):
    if hasattr(model, 'use_checkpoint'):
        model.use_checkpoint = False
    for attr in ['input_blocks', 'output_blocks']:
        if hasattr(model, attr):
            for m in getattr(model, attr):
                disable_checkpoint(m)
    if hasattr(model, 'middle_block'):
        disable_checkpoint(model.middle_block)
        
def disable_all_checkpoints(*models):
    for m in models:
        disable_checkpoint(m)
        

def paste_back(orig_img, patch_img, xy):
  
    band = 1          
    mode = 'mixed'     
    
    flags_map = {'mixed': cv2.MIXED_CLONE, 'normal': cv2.NORMAL_CLONE, 'mono': cv2.MONOCHROME_TRANSFER}
    cv2_flag = flags_map.get(mode, cv2.MIXED_CLONE)

    def pil_to_bgr_and_alpha(img: Image.Image):
        arr = np.array(img.convert('RGBA'))
        bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
        alpha = arr[..., 3]
        return bgr, alpha

    def bgr_to_pil(arr_bgr: np.ndarray):
        return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))

   
    dst_bgr, _     = pil_to_bgr_and_alpha(orig_img)
    src_bgr, src_a = pil_to_bgr_and_alpha(patch_img)
    H, W = dst_bgr.shape[:2]
    h, w = src_bgr.shape[:2]
    x, y = int(xy[0]), int(xy[1])

    
    if src_a is not None and src_a.max() > 0:
        mask_full = (src_a > 0).astype(np.uint8) * 255
    else:
        mask_full = np.full((h, w), 255, np.uint8)

    
    x0, y0, x1, y1 = x, y, x + w, y + h
    ox0, oy0 = max(0, x0), max(0, y0)
    ox1, oy1 = min(W, x1), min(H, y1)
    if ox0 >= ox1 or oy0 >= oy1:
        return orig_img  

    sx0, sy0 = ox0 - x0, oy0 - y0
    sx1, sy1 = sx0 + (ox1 - ox0), sy0 + (oy1 - oy0)

    src_c   = src_bgr[sy0:sy1, sx0:sx1].copy()
    m_fullc = mask_full[sy0:sy1, sx0:sx1].copy()

    
    base = dst_bgr.copy()
    base_roi = base[oy0:oy1, ox0:ox1]
    base_roi[m_fullc > 0] = src_c[m_fullc > 0]
    base[oy0:oy1, ox0:ox1] = base_roi  

   
    t = int(max(1, band))
    ex0, ey0 = max(0, ox0 - t), max(0, oy0 - t)
    ex1, ey1 = min(W, ox1 + t), min(H, oy1 + t)

   
    mask_ext = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
    px0, py0 = ox0 - ex0, oy0 - ey0
    ph, pw = (oy1 - oy0), (ox1 - ox0)
    mask_ext[py0:py0+ph, px0:px0+pw] = m_fullc

    ksz = min(2 * t + 1, max(3, (min(ey1 - ey0, ex1 - ex0) // 2) * 2 - 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

    inner    = cv2.erode(mask_ext, kernel)              
    ring_in  = cv2.subtract(mask_ext, inner)            
    dilated  = cv2.dilate(mask_ext, kernel)             
    ring_out = cv2.subtract(dilated, mask_ext)         
    band_ext = cv2.bitwise_or(ring_in, ring_out)       

    if np.count_nonzero(band_ext) < 10:
        band_ext = mask_ext.copy()  

   
    src_ext = base[ey0:ey1, ex0:ex1].copy()

   
    center = ((ex0 + ex1) // 2, (ey0 + ey1) // 2)
    blend_band = cv2.seamlessClone(src_ext, dst_bgr, band_ext, center, cv2_flag)

   
    final = blend_band.copy()

   
    inner_only = inner[py0:py0+ph, px0:px0+pw]      
    final_roi = final[oy0:oy1, ox0:ox1]
    final_roi[inner_only > 0] = base_roi[inner_only > 0] 
    final[oy0:oy1, ox0:ox1] = final_roi

    return bgr_to_pil(final)
   

def run(meta, config, starting_noise=None):
    
    orig_img = Image.open(meta["image"]).convert("RGB").resize((512, 512))
    meta["orig_img"] = orig_img

    USE_MANUAL_CROP = True   

    if USE_MANUAL_CROP:
        
        x, y, w, h = 192, 192, 128, 128
        crop_img = orig_img.crop((x, y, x + w, y + h))
        crop_x, crop_y = x, y
    else:
        
        crop_img, (crop_x, crop_y) = center_biased_random_crop(
            orig_img, crop_size=128, center_ratio=0.5
        )

    meta["crop_xy"] = (crop_x, crop_y)

    
    temp_crop_path = "temp_crop.png"
    crop_img.save(temp_crop_path)
    meta["image"] = temp_crop_path
    
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])

    disable_all_checkpoints(model, autoencoder, text_encoder, diffusion)

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)

    # - - - - - prepare batch - - - - - #
    meta1 = deepcopy(meta)
    meta2 = deepcopy(meta)
    meta1["prompt"] = meta["prompt1"]
    meta2["prompt"] = meta["prompt2"]

    if "edge" in meta["ckpt"]:
        batch1 = prepare_batch_edge(meta1, config.batch_size)
        batch2 = prepare_batch_edge(meta2, config.batch_size)
    else:
        batch1 = prepare_batch(meta1, config.batch_size)
        batch2 = prepare_batch(meta2, config.batch_size)

    # context encoding
    context_1 = text_encoder.encode([meta["prompt1"]] * config.batch_size)
    context_2 = text_encoder.encode([meta["prompt2"]] * config.batch_size)

    # unconditional prompt
    uc = text_encoder.encode(config.batch_size * [""])
    if args.negative_prompt is not None:
        uc = text_encoder.encode(config.batch_size * [args.negative_prompt])

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(
            diffusion, model,
            alpha_generator_func=alpha_generator_func,
            set_alpha_scale=set_alpha_scale,
            autoencoder=autoencoder,
            image_path=meta["image"],
            mask_save_path=meta['mask_save_folder'],
            defect=meta['prompt2'],
            crop_xy=meta["crop_xy"]
        )
        steps = 250
    else:
        sampler = PLMSSampler(diffusion, model)
        steps = 50

    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None
    inpainting_extra_input = None
    if "input_image" in meta:
        assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        inpainting_mask = draw_masks_from_boxes(batch1['boxes'], model.image_size).cuda()
        input_image = F.pil_to_tensor(Image.open(meta["input_image"]).convert("RGB").resize((512, 512)))
        input_image = (input_image.float().unsqueeze(0).cuda() / 255 - 0.5) / 0.5
        z0 = autoencoder.encode(input_image)
        masked_z = z0 * inpainting_mask
        inpainting_extra_input = torch.cat([masked_z, inpainting_mask], dim=1)

    # - - - - - input for gligen - - - - - #
    grounding_input_1 = grounding_tokenizer_input.prepare(batch1)
    grounding_input_2 = grounding_tokenizer_input.prepare(batch2)

    grounding_extra_input_1 = grounding_downsampler_input.prepare(batch1) if grounding_downsampler_input else None
    grounding_extra_input_2 = grounding_downsampler_input.prepare(batch2) if grounding_downsampler_input else None

    # input dict with dual prompt support
    input = dict(
        x=starting_noise,
        timesteps=None,
        context_1=context_1,
        context_2=context_2,
        grounding_input_1=grounding_input_1,
        grounding_input_2=grounding_input_2,
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input_1=grounding_extra_input_1,
        grounding_extra_input_2=grounding_extra_input_2,
    )

    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
    samples_fake = sampler.sample_New(
        S=steps,
        shape=shape,
        input=input,
        uc=uc,
        guidance_scale=config.guidance_scale,
        mask=inpainting_mask,
        x0=z0,
        context_switch_step=args.context_switch_step
    )

    samples_fake = autoencoder.decode(samples_fake)

    # - - - - - save images - - - - - #
    output_folder = os.path.join(args.folder, meta["save_folder_name"])
    os.makedirs(output_folder, exist_ok=True)
   
    start = len(os.listdir(output_folder))
    image_ids = list(range(start, start + config.batch_size))
    print("Saving image IDs:", image_ids)
  
    existing_counts = {}
    for f in os.listdir(output_folder):
        if f.endswith('.png') and '_' in f:
            prefix = f.split('_')[0]  
            existing_counts[prefix] = existing_counts.get(prefix, 0) + 1

    for sample in samples_fake:
        prompt_str = meta['prompt2'].replace(" ", "_").replace("/", "_").replace("\\", "_")
        count = existing_counts.get(prompt_str, 0) + 1
        existing_counts[prompt_str] = count
    
        img_name = f"{prompt_str}_{count}_{crop_x}_{crop_y}.png"
    
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.detach().cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
    
        orig_img = meta["orig_img"]
        crop_xy = meta["crop_xy"]
        final_img = paste_back(orig_img, sample, crop_xy)
        final_img.save(os.path.join(output_folder, img_name))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="generation_sample", help="root folder for output") 
    parser.add_argument("--images_dir", type=str,  default='./inference_img')
    parser.add_argument("--context_switch_step", type=int, default=60, help="Timestep to switch from prompt1 to prompt2")
    parser.add_argument("--save_folder_name", type=str,  default='PCB')
    parser.add_argument("--mask_save_folder", type=str, default="MASK", help="Mask image save path")
    parser.add_argument("--guidance_scale", type=float,  default=3, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead. WARNING: I did not test the code yet")
    parser.add_argument("--negative_prompt", type=str,  default=None, help="")
    args = parser.parse_args()

    files = list_files(args.images_dir)
    
    prompt_combinations = [ 
         ('normal', 'missing'),
         ('normal', 'mouse'),
         # ('normal', 'spur'),
         # ('normal', 'short'),
         # ('normal', 'open'),
         # ('normal', 'spurious')
        ]

    for p1, p2 in prompt_combinations:
        print(f"\n🔁 Reasoning Combination: {p1} → {p2}")

        meta_list = []

        for file in files:
            meta_list.append(dict(
                ckpt="./OUTPUT/pcb/tag00/edge_checkpoint_latest.pth",
                prompt1=p1,  
                prompt2=p2,  
                image=file,
                alpha_type=[0.7, 0, 0.3],
                save_folder_name=args.save_folder_name,
                mask_save_folder=os.path.join(args.folder, args.mask_save_folder),
            ))

        starting_noise = torch.randn(args.batch_size, 4, 32, 32).to(device)
        starting_noise = None
        
        for meta in meta_list:
            run(meta, args, starting_noise)
            
# python infer_PCB2_New.py --no_plms    
            
