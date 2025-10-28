import os
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import binary_fill_holes

def save_ddim_visualization(pred_vis_list, image_path, step_indices, save_folder,mask_save_path,defect,crop_xy=None):
    os.makedirs(save_folder, exist_ok=True)
    pred_vis_all = torch.cat(pred_vis_list, dim=0)  # [N, 3, H, W]
    H, W = pred_vis_all.shape[-2:]

    # 读取原图，resize，并转换为 [-1, 1] 的 tensor
    orig_pil = Image.open(image_path).convert('RGB').resize((W, H), Image.BILINEAR)
    orig_tensor = TF.to_tensor(orig_pil) * 2.0 - 1.0
    orig_tensor = orig_tensor.unsqueeze(0).repeat(pred_vis_all.shape[0], 1, 1, 1)

    def tensor_to_pil(tensor):
        tensor = tensor.detach().clamp(-1, 1)
        tensor = (tensor * 0.5 + 0.5) * 255
        array = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(array)

    pred_pil_list = [tensor_to_pil(pred_vis_all[i]) for i in range(pred_vis_all.shape[0])]

    
    # —— 在 pred_pil_list 构造完成后追加 —— 
    pred_pil_dir = os.path.join(save_folder, "pred_pil_list_frames2")
    os.makedirs(pred_pil_dir, exist_ok=True)
    
    # 展开每步的 step 索引，匹配 cat 后的一一对应关系
    expanded_steps = []
    for t, step in zip(pred_vis_list, step_indices):
        if isinstance(t, torch.Tensor):
            b = t.shape[0] if t.ndim == 4 else 1
        else:
            b = 1
        expanded_steps.extend([step] * b)
    
    # 逐张保存（严格按顺序）
    for idx, pil in enumerate(pred_pil_list):
        step_tag = expanded_steps[idx] if idx < len(expanded_steps) else idx
        pil.save(os.path.join(pred_pil_dir, f"{idx:04d}_step{step_tag:03d}.png"))

    
    orig_pil_list = [tensor_to_pil(orig_tensor[i]) for i in range(orig_tensor.shape[0])]

    # 灰度掩码
    diff_tensor = (pred_vis_all - orig_tensor).abs().detach()
    gamma = 0.5
    diff_tensor = diff_tensor.pow(gamma).clamp(0, 1)
    mask_tensor = diff_tensor.mean(dim=1, keepdim=True).pow(gamma)
    mask_tensor = mask_tensor / (mask_tensor.max() + 1e-8)
    mask_pil_list = [
        Image.fromarray((mask_tensor[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
        for i in range(mask_tensor.shape[0])
    ]

    # 二值化
    scale = 10.0
    thresholds = [0.7, 0.8, 0.9, 0.93]
    sigmoid_tensor = torch.sigmoid((mask_tensor - 0.5) * scale)
    binary_pil_dict = {}
    for threshold in thresholds:
        binary_tensor = (sigmoid_tensor > threshold).float()
        binary_pil_list = [
            Image.fromarray((binary_tensor[i, 0].cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
            for i in range(binary_tensor.shape[0])
        ]
        binary_pil_dict[threshold] = binary_pil_list

    # ==================== 新增：每一步分别处理三列 ====================
    region_list = []
    erode_list = []
    overlay_list = []
    blurred_list = []

    orig_gray_np = np.array(orig_pil.convert('L'))
    thresh_value = 90
    kernel = np.ones((3, 3), np.uint8)

    # 可调参数
    expand_dist = 5  # 羽化范围，越大模糊越宽
    alpha_power = 2.0  # 羽化边缘衰减指数，越大边缘越深

    for i, pred_pil in enumerate(pred_pil_list):
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

        overlay = cv2.cvtColor(orig_gray_np, cv2.COLOR_GRAY2RGB)
        overlay[largest_mask > 0] = [255, 0, 0]

        region_list.append(Image.fromarray(largest_mask).convert("RGB"))
        erode_list.append(Image.fromarray(mask_eroded).convert("RGB"))
        overlay_list.append(Image.fromarray(overlay))

        # # 扩张羽化模糊
        # === 羽化操作 ===
        mask_inv = cv2.bitwise_not(largest_mask)
        black_region = (mask_inv == 0).astype(np.uint8)

        # 膨胀出羽化边缘范围
        feather_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand_dist + 1, 2 * expand_dist + 1))
        expanded = cv2.dilate(black_region, feather_kernel, iterations=1)
        feather_mask = (expanded > 0) & (black_region == 0)

        # 距离变换构造 alpha 衰减
        dist_map = cv2.distanceTransform(1 - black_region, cv2.DIST_L2, 5)
        alpha = np.zeros_like(dist_map, dtype=np.float32)
        alpha[feather_mask] = 1.0 - np.clip(dist_map[feather_mask] / expand_dist, 0, 1)
        alpha = alpha ** alpha_power

        # 生成羽化结果图
        result = np.ones_like(mask_inv, dtype=np.float32)
        result[black_region == 1] = 0
        result[feather_mask] = 1.0 - alpha[feather_mask]

        result_img = (result * 255).astype(np.uint8)
        blurred_list.append(Image.fromarray(result_img).convert("L"))

        # # === 整个黑色掩码内由内向外羽化 ===
        # mask_inv = cv2.bitwise_not(largest_mask)  # 黑=0, 白=255
        # black_region = (mask_inv == 0).astype(np.uint8)  # 1=黑掩码，0=其它
        #
        # # 1. 计算每个黑区像素到“边缘”的距离（即内到外距离）
        # dist_map = cv2.distanceTransform(black_region, cv2.DIST_L2, 5)
        #
        # # 2. 距离归一化&控制羽化带宽度
        # alpha = np.ones_like(dist_map, dtype=np.float32)  # 默认外部全白
        # inside = (black_region == 1)
        # # 距离为0（黑色边界）是最亮，往内逐渐变黑（梯度由 expand_dist 控制带宽）
        # alpha[inside] = np.clip(dist_map[inside] / expand_dist, 0, 1)
        # alpha[inside] = alpha[inside] ** alpha_power  # 曲线可调
        #
        # # 3. 掩码中心为0，边界为1，外部为1
        # result = np.ones_like(mask_inv, dtype=np.float32)
        # result[inside] = 1.0 - alpha[inside]  # 内部：中心黑、边界白，外部=1
        #
        # result_img = (result * 255).astype(np.uint8)
        # blurred_list.append(Image.fromarray(result_img).convert("L"))

        # # ✅ 只在处理最后一张 pred_pil 后才保存
        # if i == len(pred_pil_list) - 1 and mask_save_path is not None and len(region_list) > 0:
        #     os.makedirs(mask_save_path, exist_ok=True)
        #     existing_files = [
        #         f for f in os.listdir(mask_save_path)
        #         if f.startswith(defect + "_") and f.endswith(".png")
        #     ]
        #     index = len(existing_files) + 1
        #     save_name = f"{defect}_{index}.png"
        #     save_path = os.path.join(mask_save_path, save_name)
        #     region_list[-1].save(save_path)
        if i == len(pred_pil_list) - 1 and mask_save_path is not None and len(region_list) > 0:
            os.makedirs(mask_save_path, exist_ok=True)
            existing_files = [
                f for f in os.listdir(mask_save_path)
                if f.startswith(defect + "_") and f.endswith(".png")
            ]
            index = len(existing_files) + 1
            save_name = f"{defect}_{index}.png"
            save_path = os.path.join(mask_save_path, save_name)

            # 拼接512x512大蒙版
            crop_x, crop_y = crop_xy  # crop_xy传入进来
            mask_big = Image.new("L", (512, 512), 0)
            mask_small = region_list[-1].convert("L").resize((128, 128))
            mask_big.paste(mask_small, (crop_x, crop_y))
            mask_big.save(save_path)

    # ==================== 拼接可视化相关函数 ====================
    def hconcat_images(img_list, padding=0, pad_color=(255, 255, 255)):
        widths, heights = zip(*(im.size for im in img_list))
        total_width = sum(widths) + padding * (len(img_list) - 1)
        max_height = max(heights)
        mode = img_list[0].mode
        if mode == "L":
            pad_val = pad_color[0] if isinstance(pad_color, (tuple, list)) else pad_color
            new_im = Image.new(mode, (total_width, max_height), pad_val)
        else:
            new_im = Image.new(mode, (total_width, max_height), pad_color)
        x_offset = 0
        for im in img_list:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0] + padding
        return new_im

    def vconcat_images(img_list, padding=0, pad_color=(255, 255, 255)):
        widths, heights = zip(*(im.size for im in img_list))
        total_height = sum(heights) + padding * (len(img_list) - 1)
        max_width = max(widths)
        mode = img_list[0].mode
        if mode == "L":
            pad_val = pad_color[0] if isinstance(pad_color, (tuple, list)) else pad_color
            new_im = Image.new(mode, (max_width, total_height), pad_val)
        else:
            new_im = Image.new(mode, (max_width, total_height), pad_color)
        y_offset = 0
        for im in img_list:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1] + padding
        return new_im

    def add_column_title(image, title, font_size=14, pad=40):
        width, height = image.size
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        new_img = Image.new("RGB", (width, height + pad), color=(255, 255, 255))
        new_img.paste(image, (0, pad))
        draw = ImageDraw.Draw(new_img)
        text_width, _ = draw.textsize(title, font=font)
        draw.text(((width - text_width) // 2, 5), title, font=font, fill=(0, 0, 0))
        return new_img

    # Step标签列
    labeled_pred_list = []
    label_width = 90
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    for i, im in enumerate(pred_pil_list):
        label_img = Image.new("RGB", (label_width, im.height), (255, 255, 255))
        draw = ImageDraw.Draw(label_img)
        text = f"img" if i == len(step_indices) - 1 else f"Step {step_indices[i]}"
        text_width, text_height = draw.textsize(text, font=font)
        draw.text(((label_width - text_width) // 2, (im.height - text_height) // 2),
                  text, font=font, fill=(0, 0, 0))
        combined = hconcat_images([label_img, im], padding=0)
        labeled_pred_list.append(combined)

    pred_col = vconcat_images(labeled_pred_list, padding=5)
    pred_col = add_column_title(pred_col, "Pred x0")
    orig_col = vconcat_images(orig_pil_list, padding=5)
    orig_col = add_column_title(orig_col, "original")
    mask_col = vconcat_images(mask_pil_list, padding=5)
    mask_col = add_column_title(mask_col, "gray")
    binary_cols = []
    for threshold in thresholds:
        col = vconcat_images(binary_pil_dict[threshold], padding=5)
        col = add_column_title(col, f"binary_{threshold}")
        binary_cols.append(col)

    # 新增三列
    region_col = vconcat_images(region_list, padding=5)
    region_col = add_column_title(region_col, "Largest Region")
    erode_col = vconcat_images(erode_list, padding=5)
    erode_col = add_column_title(erode_col, "Eroded")
    overlay_col = vconcat_images(overlay_list, padding=5)
    overlay_col = add_column_title(overlay_col, "Overlay")
    blur_col = vconcat_images(blurred_list, padding=5)
    blur_col = add_column_title(blur_col, f"Invert+EdgeBlur({expand_dist}px)")

    # 拼接所有列
    final_img = hconcat_images(
        [pred_col, orig_col, mask_col] + binary_cols +
        [ erode_col,region_col, overlay_col, blur_col],  # 多了blur_col
        padding=5
    )

    # 自动编号保存
    existing_files = [f for f in os.listdir(save_folder) if f.startswith("combined_result_") and f.endswith(".png")]
    indices = [int(f[16:-4]) for f in existing_files if f[16:-4].isdigit()]
    next_index = max(indices) + 1 if indices else 1
    save_path = os.path.join(save_folder, f"combined_result_{next_index:03d}.png")
    final_img.save(save_path)
    print(f"✅ 拼接图保存到 {save_path}")

