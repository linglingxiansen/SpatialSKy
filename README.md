# Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation
Repository for **Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation**


![image](image/image.png)


<h5 align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-9BDFDF)](https://github.com/linglingxiansen/SpatialSky/blob/main/LICENSE) 
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoint-FBD49F.svg)](https://huggingface.co/llxs/Sky-VLM)
[![arXiv](https://img.shields.io/badge/Arxiv-2511.13269-E69191.svg?logo=arXiv)](https://arxiv.org/abs/2511.13269) 


## 🚀 Getting Start
### Installation
```
git clone https://github.com/linglingxiansen/SpatialSKy.git
pip install git+https://github.com/huggingface/transformers accelerate torch torchvision openai pillow tqdm nltk scipy
```
### Scene Dataset
Download the uav scene dataset from [UAVScenes](https://github.com/sijieaaa/UAVScenes) and put the UAVScenes dir in SpatialSky. The file structure should look like this:
```
SpatialSKy/
├── UAVScenes
|   ├── interval5_CAM_LIDAR
|   ├── interval5_CAM_label
|   ├── interval5_LIDAR_label
|   └──...
├── benchmark
├── metric
└── parallel_inference.py
```

### Download Ckpt

Download our Sky-VLM model from [![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoint-FBD49F.svg)](https://huggingface.co/llxs/Sky-VLM).


## 🔍 Evaluation for SpatialSky-Bench
For inference:
```
python parallel_inference.py --ckpt_path /your/ckpt/path --num_gpus $your GPU number
```

For metrics computing:
```
cd metric
bash eval.sh /your/result/dir
```
## 🔥 Training
The training dataset and code are coming soon...


## Citation
If this work is helpful, please kindly cite as:
```
@article{spatialsky,
      title={Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation}, 
      author={Zhang, Lingfeng and Zhang, Yuchen and Li, Hongsheng and Fu, Haoxiang and Tang, Yingbo and Ye, Hangjun and Chen, Long and Liang, Xiaojun and Hao, Xiaoshuai and Ding, Wenbo},
      journal={arXiv preprint arXiv:2511.13269},
      year={2025},
}
```




