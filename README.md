# PatchDefDiffusion
Authentic Defect Image Generation via Local Patch Synthesis with Structure Guidance and Non-Defect Area Reconstruction
<img width="4426" height="1277" alt="图片1" src="https://github.com/user-attachments/assets/eb9d1fe1-1ed9-45d3-9627-f600bba2efc6" />
# Dataset
1.Download the dataset https://robotics.pkusz.edu.cn/resources/dataset/
2.Crop the defect images to the configured PATCH_SIZE. Using identical coordinates and dimensions, crop the corresponding regions from the paired normal images and extract the edge maps for those regions. Name both outputs using the convention <defect_name>_<index>. Place the edge maps in ".DATA/PCB2/dataset/train/Ground_truth" and the defect images in ".DATA/PCB2/dataset/train/Source_Images".
<img width="1232" height="744" alt="图片2" src="https://github.com/user-attachments/assets/a344f101-7393-4243-bfde-ce3ccbb5e278" />
3.Generate a "caption.json" file for the dataset by referring to "process_json_pcb.py".
