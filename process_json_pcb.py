import os
import json

input_folder = './DATA/PCB2/dataset/train/Source_Images'  # 替换为你的文件夹路径
output_file = './DATA/PCB2/dataset/train/caption.json'  # 输出的 JSON 文件名

file_names = os.listdir(input_folder)

data = []


for file_name in file_names:
    file_dict = {}
    if os.path.isfile(os.path.join(input_folder, file_name)):
        file_name_sub = file_name.split('.')[0]
        last_underscore_index = file_name_sub.rfind("_")
        file_name_sub = file_name_sub[:last_underscore_index]
        if file_name_sub == "missing":
            file_dict[file_name] = f"A photo of missing"
        elif file_name_sub == "mouse":
            file_dict[file_name] = f"A photo of mouse"
        elif file_name_sub == "open":
            file_dict[file_name] = f"A photo of open"
        elif file_name_sub == "short":
            file_dict[file_name] = f"A photo of short"
        elif file_name_sub == "spur":
            file_dict[file_name] = f"A photo of spur"
        elif file_name_sub == "spurious":
            file_dict[file_name] = f"A photo of spurious"
        elif file_name_sub == "normal":
            file_dict[file_name] = f"A photo of normal"
        data.append(file_dict)

with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print("Create JSON successfully!")
