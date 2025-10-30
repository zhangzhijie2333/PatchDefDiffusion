# PatchDefDiffusion
Authentic Defect Image Generation via Local Patch Synthesis with Structure Guidance and Non-Defect Area Reconstruction
<img width="4426" height="1277" alt="1" src="https://github.com/user-attachments/assets/eb9d1fe1-1ed9-45d3-9627-f600bba2efc6" />
## Dataset
1.Download the dataset [PKU-Market-PCB](https://robotics.pkusz.edu.cn/resources/dataset/)  
2.Crop the defect images to the configured PATCH_SIZE. Using identical coordinates and dimensions, crop the corresponding regions from the paired normal images and extract the edge maps for those regions. Name both outputs using the convention <defect_name>_<index>. Place the edge maps in ".DATA/PCB2/dataset/train/Ground_truth" and the defect images in ".DATA/PCB2/dataset/train/Source_Images".
<p align="center">
 <img width="428" height="259" alt="2" src="https://github.com/user-attachments/assets/dc5c4d3b-c4c9-44fe-a67d-cff2930002cb" />
</p>  
3.Generate a "caption.json" file for the dataset by referring to "process_json_pcb.py". 

## Checkpoints
All the checkpoints can be downloaded from the following links. And some checkpoints should be placed at the corresponding directory.
| Data and Models                          | Download                                                                                                    | Place at                                 |
|------------------------------------------|-----------------------------------------                                                                    |----------------------------------        |
| Checkpoints for defect generation model  | [Google Drive](https://drive.google.com/drive/folders/1SXWqeQsvFmXdcNOugFKvp17Q4VLPnIYW?usp=drive_link)     | Place in `OUTPUT`                        |
| Checkpoints for models of SD-V-1.4       | [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt)       | Place in the root directory              |
| Checkpoints for  models of clip          | [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)                              | Place in `openai/clip-vit-large-patch14` |

## Prepare
### (1) Prepare the environment
```bash
# Install dependencies (requirements)
pip install -r requirements.txt
```
### (2) Checkpoint for SD-V-1.4 and clip 

### (3) Prepare Dataset and Generate a "caption.json" file
```bash
 python process_json_pcb.py
```
## Train the Patch-level defect model
```bash
 python main.py
```
## Inference
Before performing inference, please place the reference images into the "./inference_img/" folder in the project root directory.
```bash
  python infer_PCB2_New.py --no_plms    
```
The defect location can be manually specified.
<p align="center">
 <img width="422" height="73" alt="image" src="https://github.com/user-attachments/assets/8bce45f8-acf6-4d59-b946-70624b3c939f" />
</p> 

## Citation
If you make use of our work, please cite our paper:
```bash

```
