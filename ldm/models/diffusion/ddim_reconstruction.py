import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import yaml
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import os
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import math
import os
from datetime import datetime
import gc
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

device = torch.device("cuda") 


class DDIMSampler(object):
    def __init__(self, diffusion, model, schedule="linear", alpha_generator_func=None, set_alpha_scale=None,
                 autoencoder = None, image_path=None,mask_save_path=None,defect=None,crop_xy=None):
        super().__init__()
        self.alphas_cumprod_prev = None
        self.diffusion = diffusion
        self.model = model
        self.device =device
        self.ddpm_num_timesteps = diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale

        self.autoencoder = autoencoder

        self.mask_save_path = mask_save_path
        self.image_path = image_path
        self.defect = defect
        
        self.crop_xy = crop_xy
               
        with open("configs/optimize_configs/xt_optim.yaml", "r") as f:
              xt_opt_config = yaml.safe_load(f)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        xt_opt_config["run_id"] = run_id
        self.xt_opt_config = xt_opt_config        
        cfg = self.xt_opt_config or {}
        self.optimize_xt = cfg.get("optimize_xt", True)
        self.optimize_xt_interval_list = cfg.get("optimize_xt_interval_list", None)
        self.num_iteration_optimize_xt = cfg.get("num_iteration_optimize_xt", 4)
        self.lr_xt = cfg.get("lr_xt", 0.02)
        self.use_adaptive_lr_xt = cfg.get("use_adaptive_lr_xt", True)
        self.coef_xt_reg = cfg.get("coef_xt_reg", 0.0001)
        self.use_smart_lr_xt_decay = cfg.get("use_smart_lr_xt_decay", False)
        self.mid_interval_num = cfg.get("mid_interval_num", 1)
        self.lr_xt_decay = cfg.get("lr_xt_decay", 1.012)
        self.coef_xt_reg_decay = cfg.get("coef_xt_reg_decay", 1.01)
        self.use_mask_interval_list = cfg.get("use_mask_interval_list",None)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=False)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        # import pdb;pdb.set_trace()
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=False)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def sample_New(self, S, shape, input, uc=None, guidance_scale=1, mask=None, x0=None, context_switch_step=None):
        self.make_schedule(ddim_num_steps=S)

        return self.ddim_sampling_New(
            shape=shape,
            input=input,
            uc=uc,
            guidance_scale=guidance_scale,
            mask=mask,
            x0=x0,
            context_switch_step=context_switch_step 
        )

    def ddim_sampling_New(self, shape, input, uc, guidance_scale=1, mask=None, x0=None,
                      context_switch_step=None):

        autoencoder = self.autoencoder
        image_path = self.image_path
    
        b = shape[0]
        img = input["x"]
        if img is None:
            img = torch.randn(shape, device=self.device)
            input["x"] = img
    
        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        iterator = time_range
    
        if self.alpha_generator_func is not None:
            alphas = self.alpha_generator_func(len(iterator))
    
        lr_xt = self.lr_xt
        coef_xt_reg = self.coef_xt_reg
    
        pbar = tqdm(total=total_steps, desc="sampling", unit="step", leave=False)
        for i, step in enumerate(iterator):
            if self.alpha_generator_func is not None:
                self.set_alpha_scale(self.model, alphas[i])
    
            index = total_steps - i - 1
            input["timesteps"] = torch.full((b,), step, device=self.device, dtype=torch.long)
    
            if context_switch_step is not None and i > context_switch_step:
                input["context"] = input["context_2"]
                input["grounding_input"] = input.get("grounding_input_2", None)
                input["grounding_extra_input"] = input.get("grounding_extra_input_2", None)
            else:
                input["context"] = input["context_1"]
                input["grounding_input"] = input.get("grounding_input_1", None)
                input["grounding_extra_input"] = input.get("grounding_extra_input_1", None)
    
            if mask is not None:
                assert x0 is not None
                img_orig = self.diffusion.q_sample(x0, input["timesteps"])
                img = img_orig * mask + (1. - mask) * img
                input["x"] = img
    
            use_xt_opt = False
            if self.optimize_xt_interval_list is not None:
                for start, end in self.optimize_xt_interval_list:
                    if start < i < end:
                        use_xt_opt = True
                        break
    
            if use_xt_opt:
                img, pred_x0 = self.p_sample_ddim_New_mask(
                    input, index=index, uc=uc, guidance_scale=guidance_scale,
                    sampling_step_index=i, lr_xt=lr_xt, coef_xt_reg=coef_xt_reg,
                )
                torch.cuda.empty_cache()
                gc.collect()
                lr_xt *= self.lr_xt_decay
                coef_xt_reg *= self.coef_xt_reg_decay
            else:
                img, pred_x0 = self.p_sample_ddim(
                    input, index=index, uc=uc, guidance_scale=guidance_scale
                )
                torch.cuda.empty_cache()
                gc.collect()
    
            input["x"] = img
            pbar.update(1)
        pbar.close()
    
        return img

    @torch.no_grad()
    def p_sample_ddim(self, input, index, uc=None, guidance_scale=1):

        e_t = self.model(input)
        if uc is not None and guidance_scale != 1:
            unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc, inpainting_extra_input=input["inpainting_extra_input"], grounding_extra_input=input['grounding_extra_input'])
            e_t_uncond = self.model( unconditional_input )
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        # select parameters corresponding to the currently considered timestep
        b = input["x"].shape[0]
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)

        # current prediction for x_0
        pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like( input["x"] )
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

     
    def get_smart_lr_decay_rate(self, _t, interval_num):
        int_t = int(_t[0].item())
        interval = int_t // interval_num

        steps = (
            (np.arange(0, interval_num) * interval)
            .round()[::-1]
            .copy()
            .astype(np.int32)
        )
        steps = steps.tolist()
        if steps[0] != int_t:
            steps.insert(0, int_t)
        if steps[-1] != 0:
            steps.append(0)

        ret = 1
        time_pairs = list(zip(steps[:-1], steps[1:]))
        for i in range(len(time_pairs)):
            _cur_t, _prev_t = time_pairs[i]
            ret *= self.sqrt_recip_alphas_cumprod[_cur_t] * math.sqrt(
                self.alphas_cumprod[_prev_t]
            )
        return ret

    def p_sample_ddim_New_mask(
            self,
            input,
            index,
            uc=None,
            guidance_scale=1,
            sampling_step_index=None,
            lr_xt=0.02,
            coef_xt_reg=0.0001,
    ):
        e_t = self.model(input)

        if uc is not None and guidance_scale != 1:
            unconditional_input = dict(
                x=input["x"],
                timesteps=input["timesteps"],
                context=uc,
                inpainting_extra_input=input["inpainting_extra_input"],
                grounding_extra_input=input['grounding_extra_input']
            )
            e_t_uncond = self.model(unconditional_input)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        # select parameters corresponding to the currently considered timestep
        b = input["x"].shape[0]
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=self.device)

        # ========= copaint_optimize_xt_and_recompute_xprev  =========
        
        def loss_fn(_x0, _pred_x0, _mask=None):
            
            diff = _x0 - _pred_x0
            if _mask is not None:
              
                if _mask.ndim == 2:
                    _mask = _mask.unsqueeze(0) 
               
                diff = diff * _mask
            return torch.sum(diff ** 2)

        def reg_fn(_origin_xt, _xt):
            ret = torch.sum((_origin_xt - _xt) ** 2)
            return ret

        def gen_mask(orig_pil, pred_pil, thresh_value=100):
            
            kernel = np.ones((3, 3), np.uint8)
            orig_gray_np = np.array(orig_pil.convert('L'))
            pred_gray_np = np.array(pred_pil.convert('L'))
            
            _, ssim_map = ssim(pred_gray_np, orig_gray_np, full=True)
            diff = (1 - ssim_map) * 255
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            
            _, mask = cv2.threshold(diff, thresh_value, 255, cv2.THRESH_BINARY)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask_filled = binary_fill_holes(mask_clean > 0).astype(np.uint8) * 255
                   
            mask_eroded = cv2.erode(mask_filled, kernel, iterations=1)
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_eroded, connectivity=8)
            if num_labels > 1:
                max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                largest_mask = np.zeros_like(mask_eroded)
                largest_mask[labels == max_label] = 255
            else:
                largest_mask = mask_eroded.copy()
        
            mask_inv = cv2.bitwise_not(largest_mask)
            mask_inv_norm = mask_inv.astype(np.float32) / 255.0
            largest_mask_pil = Image.fromarray(largest_mask).convert("RGB")
            return largest_mask_pil, mask_inv_norm

        # ==================
        t_index = sampling_step_index
        if isinstance(t_index, torch.Tensor):
            t_index = t_index.item()

        # ==================
        if self.use_smart_lr_xt_decay:
            decay_rate = self.get_smart_lr_decay_rate(
                _t=input['timesteps'],
                interval_num=self.mid_interval_num
            )
            lr_xt /= decay_rate

       
        original_img = Image.open(self.image_path)  
      
        img = np.array(original_img).astype(np.float32) / 255.0  
        img = img * 2.0 - 1.0  
        img = np.transpose(img, (2, 0, 1)) 
        gt_tensor = torch.from_numpy(img).to(self.device)  


        with torch.enable_grad():
            # ==================
            origin_xt = input["x"].clone().detach()
            x = input["x"].clone().detach().to(self.device).requires_grad_()
            lr = lr_xt

            e_t = self.model(input)
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            decoded = self.autoencoder.decode(pred_x0)

            sample = torch.clamp(decoded, min=-1, max=1) * 0.5 + 0.5
            sample_np = sample.detach().cpu().numpy()
            if sample_np.ndim == 4:
                sample_np = sample_np[0]
            if sample_np.ndim == 3 and sample_np.shape[0] in [1, 3]:
                sample_np = sample_np.transpose(1, 2, 0)
            sample_img = np.clip(sample_np * 255, 0, 255).astype(np.uint8)
            sample = Image.fromarray(sample_img)


            in_mask_interval = any(
                start < sampling_step_index < end
                for start, end in self.use_mask_interval_list
            )


            batch_size, channels, h, w = input["x"].shape
            mask = None
            if in_mask_interval:
                _, mask = gen_mask(original_img, sample)
            z_mask = None  
            if mask is not None:
               
                if not torch.is_tensor(mask):
                    mask = torch.from_numpy(mask).float().to(gt_tensor.device)
             
                if mask.max() > 1.0:
                    mask = mask / 255.0
             
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)

                mask_batched = mask.expand(batch_size, -1, mask.shape[-2], mask.shape[-1])  
             
                mask_down = F.interpolate(mask_batched, size=(h, w), mode="nearest") 
              
                z_mask = mask_down.expand(-1, channels, -1, -1).contiguous() 
            
            init_loss = loss_fn(decoded, gt_tensor, mask).item()

            for step in range(self.num_iteration_optimize_xt):
                loss = loss_fn(gt_tensor, decoded, mask) + coef_xt_reg * reg_fn(origin_xt, x)  

                x_grad = torch.autograd.grad(
                    loss, x, retain_graph=False, create_graph=False
                )[0].detach()
                if z_mask is not None:
                    new_x = x - lr_xt * x_grad * z_mask
                else:
                    new_x = x - lr_xt * x_grad


                while self.use_adaptive_lr_xt and True:
                    with torch.no_grad():
                        input['x'] = new_x
                        e_t = self.model(input)
                        pred_x0 = (new_x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                        decoded = self.autoencoder.decode(pred_x0)
                        new_loss = loss_fn(gt_tensor, decoded, mask) + coef_xt_reg * reg_fn(origin_xt, new_x)

                        if not torch.isnan(new_loss) and new_loss <= loss:
                            break
                        else:
                            lr_xt *= 0.8
                            del new_x, e_t, pred_x0, new_loss, input['x'], decoded
                            if z_mask is not None:
                                new_x = x - lr_xt * x_grad * z_mask
                            else:
                                new_x = x - lr_xt * x_grad

                x = new_x.detach().requires_grad_()
                input['x'] = x
                e_t = self.model(input)
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                decoded = self.autoencoder.decode(pred_x0)

                del loss, x_grad
                torch.cuda.empty_cache()

        # after optimize
        with torch.no_grad():
            new_loss = loss_fn(gt_tensor, decoded, mask).item()
            new_reg = reg_fn(origin_xt, new_x).item()
            pred_x0, e_t, x = pred_x0.detach(), e_t.detach(), x.detach()
            del origin_xt, init_loss

            # ==================
            with torch.no_grad():
                final_pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
                noise = sigma_t * torch.randn_like(x)
                x_prev = a_prev.sqrt() * final_pred_x0 + dir_xt + noise

                if (index == 0) and (getattr(self, "mask_save_path", None) is not None) and (getattr(self, "crop_xy", None) is not None):
                    os.makedirs(self.mask_save_path, exist_ok=True)
                    defect_prefix = getattr(self, "defect", "mask")
                    existing_files = [f for f in os.listdir(self.mask_save_path) if f.startswith(defect_prefix + "_") and f.endswith(".png")]
                    idx = len(existing_files) + 1
                    save_name = f"{defect_prefix}_{idx}.png"
                    save_path = os.path.join(self.mask_save_path, save_name)
                
                    sample_save = torch.clamp(self.autoencoder.decode(final_pred_x0), min=-1, max=1) * 0.5 + 0.5
                    sample_np = sample_save.detach().cpu().numpy()
                    if sample_np.ndim == 4:
                        sample_np = sample_np[0]
                    if sample_np.ndim == 3 and sample_np.shape[0] in [1, 3]:
                        sample_np = sample_np.transpose(1, 2, 0)
                    sample_img = np.clip(sample_np * 255, 0, 255).astype(np.uint8)
                    sample_pil = Image.fromarray(sample_img)
                
                    largest_mask_pil, _ = gen_mask(original_img, sample_pil)
                    crop_x, crop_y = self.crop_xy
                    mask_big = Image.new("L", (512, 512), 0)
                    mask_small = largest_mask_pil.convert("L").resize((128, 128))
                    mask_big.paste(mask_small, (crop_x, crop_y))
                    mask_big.save(save_path)
                    
                # ============
                del x, decoded, pred_x0, e_t, new_x, dir_xt, noise,gt_tensor
                gc.collect()
                torch.cuda.empty_cache()

        return x_prev.detach(), final_pred_x0.detach()















