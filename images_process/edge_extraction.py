from tqdm import tqdm
import cv2
import numpy as np
import os

def smooth_pcb_edge_extraction(image_path, save_folder):
    """
    使用 CLAHE + 高斯模糊 + 自适应 Canny + 闭运算 提取平滑的 PCB 边缘图像。
    保存到 save_folder 下，文件名加上 _edge 后缀。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片读取失败：{image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE 对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 高斯模糊
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.4)

    # 自适应 Canny 边缘检测
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)

    # 闭运算（连接断裂边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.basename(image_path)
    name, _ = os.path.splitext(base_name)

    # 寻找一个不重复的编号
    i = 1
    while True:
        save_name = f"{i:03d}_{name}_edge.png"
        save_path = os.path.join(save_folder, save_name)
        if not os.path.exists(save_path):
            break
        i += 1

    cv2.imwrite(save_path, closed)

    return save_path


def batch_smooth_pcb_edges(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
    ]
    for img_name in tqdm(image_files, desc="批量提取顺滑边缘图"):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + '.png')
        smooth_pcb_edge_extraction(input_path, output_path)
    return output_folder

def smooth_pcb_edge_extraction(image_path):
    """
    使用 CLAHE + 高斯模糊 + 自适应 Canny + 闭运算 提取平滑的 PCB 边缘图像。
    返回边缘图 ndarray（uint8, 0/255）。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"图片读取失败：{image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.4)
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def batch_smooth_pcb_edges(input_folder):
    image_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
    ]
    results = []
    for img_name in tqdm(image_files, desc="批量提取顺滑边缘图"):
        input_path = os.path.join(input_folder, img_name)
        edge = smooth_pcb_edge_extraction(input_path)
        results.append(edge)
    return results

