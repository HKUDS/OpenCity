# OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction

<img src='opencity.png' />

A pytorch implementation for the paper: [OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction](https://arxiv.org/abs/2408.10269)<br />  

[Zhonghang Li](https://scholar.google.com/citations?user=__9uvQkAAAAJ), [Long Xia](https://scholar.google.com/citations?user=NRwerBAAAAAJ), [Lei Shi](https://harryshil.github.io/), [Yong Xu](https://scholar.google.com/citations?user=1hx5iwEAAAAJ), [Dawei Yin](https://www.yindawei.com/), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />  

**[Data Intelligence Lab](https://sites.google.com/view/chaoh/home)@[University of Hong Kong](https://www.hku.hk/)**, [South China University of Technology](https://www.scut.edu.cn/en/), Baidu Inc  
<!--
-----

<a href='https://OpenCity-ST.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://github.com/HKUDS/OpenCity'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<#><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=4BIbQt-EIAM)
 • 🌐 <a href="https://zhuanlan.zhihu.com/p/684785925" target="_blank">中文博客</a>
-->
This repository hosts the code, data, and model weights of **OpenCity**.

-----
## 🎉 News 
- [x] [2024.08.21] Release the full paper.
- [x] [2024.08.20] Add video.
- [x] [2024.08.15] 🚀🚀 Release the code, model weights and datasets of OpenCity.
- [x] [2024.08.15] Release baselines codes.


🎯🎯📢📢 We upload the **models** and **data** used in our OpenCity on 🤗 **Huggingface**. We highly recommend referring to the table below for further details: 

| 🤗 Huggingface Address                                        | 🎯 Description                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [https://huggingface.co/hkuds/OpenCity-Plus](https://huggingface.co/hkuds/OpenCity-Plus/tree/main) | It's the model weights of our OpenCity-Plus. |
| [https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main](https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main) | We released the datasets used in OpenCity. |

## 👉 TODO 
...


-----------

## Introduction

<p style="text-align: justify">
In this work, we aim to unlock new possibilities for building versatile, resilient and adaptive spatio-temporal foundation models for traffic prediction. 
To achieve this goal, we introduce a novel foundation model, named OpenCity, that can effectively capture and normalize the underlying spatio-temporal patterns from diverse data characteristics, facilitating zero-shot generalization across diverse urban environments. 
OpenCity integrates the Transformer architecture with graph neural networks to model the complex spatio-temporal dependencies in traffic data. 
By pre-training OpenCity on large-scale, heterogeneous traffic datasets, we enable the model to learn rich, generalizable representations that can be seamlessly applied to a wide range of traffic forecasting scenarios. 
Experimental results demonstrate that OpenCity exhibits exceptional zero-shot predictive performance in various traffic prediction tasks.
</p>

![The detailed framework of the proposed OpenCity.](https://github.com/OpenCity-ST/OpenCity-ST.github.io/blob/main/images/framework.png)

## Main Results
**Outstanding Zero-shot Prediction Performance.** OpenCity achieves significant zero-shot learning breakthroughs, outperforming most baselines even without fine-tuning. This highlights the approach's robustness and effectiveness at learning complex spatio-temporal patterns in large-scale traffic data, extracting universal insights applicable across downstream tasks.
 
![Zero-shot vs. Full-shot.](https://github.com/OpenCity-ST/OpenCity-ST.github.io/blob/main/images/zero-shot.png)



### Demo Video
https://github.com/user-attachments/assets/39265dc5-0126-483b-951e-518c6cb210e0

-----------
<span id='Usage'/>

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment'>2. Environment </a>
* <a href='#Training OpenCity'>3. Training OpenCity </a>
  * <a href='#Preparing Pre-trained Data'>3.1. Preparing Pre-trained Data </a>
  * <a href='#Pre-training'>3.2. Pre-training </a>
* <a href='#Evaluating'>4. Evaluating </a>
****


<span id='Code Structure'/>

### 1. Code Structure <a href='#all_catelogue'>[Back to Top]</a>

```
├── conf/
│   ├── AGCRN/
│   │   └── AGCRN.conf
│   ├── ASTGCN/
│   │   └── ASTGCN.conf
│   ├── general_conf/
│   │   ├── global_baselines.conf
│   │   └── pretrain.conf
│   ├── GWN/
│   │   └── GWN.conf
│   ├── MSDR/
│   │   └── MSDR.conf
│   ├── MTGNN/
│   │   └── MTGNN.conf
│   ├── OpenCity/
│   │   └── OpenCity.conf
│   ├── PDFormer/
│   │   └── PDFormer.conf
│   ├── STGCN/
│   │   └── STGCN.conf
│   ├── STSGCN/
│   │   └── STSGCN.conf
│   ├── STWA/
│   │   └── STWA.conf
│   └── TGCN/
│       └── TGCN.conf
├── data/
│   ├── generate_ca_data.py
│   └── README.md
├── lib/
│   ├── data_process.py
│   ├── logger.py
│   ├── metrics.py
│   ├── Params_predictor.py
│   ├── Params_pretrain.py
│   ├── predifineGraph.py
│   └── TrainInits.py
├── model/
│   ├── AGCRN/
│   │   ├── AGCN.py
│   │   ├── AGCRN.py
│   │   ├── AGCRNCell.py
│   │   └── args.py
│   ├── ASTGCN/
│   │   ├── args.py
│   │   └── ASTGCN.py
│   ├── GWN/
│   │   ├── args.py
│   │   └── GWN.py
│   ├── MSDR/
│   │   ├── args.py
│   │   ├── gmsdr_cell.py
│   │   └── gmsdr_model.py
│   ├── MTGNN/
│   │   ├── args.py
│   │   └── MTGNN.py
│   ├── OpenCity/
│   │   ├── args.py
│   │   └── OpenCity.py
│   ├── PDFormer/
│   │   ├── args.py
│   │   └── PDFormer.py
│   ├── ST_WA/
│   │   ├── args.py
│   │   ├── attention.py
│   │   └── ST_WA.py
│   ├── STGCN/
│   │   ├── args.py
│   │   └── stgcn.py
│   ├── STSGCN/
│   │   ├── args.py
│   │   └── STSGCN.py
│   └── TGCN/
│       ├── args.py
│       └── TGCN.py
│   ├── Model.py
│   ├── BasicTrainer.py
│   ├── Run.py
└── model_weights/
    ├── OpenCity/
    └── README.md
```


<span id='Environment'/>

### 2.Environment <a href='#all_catelogue'>[Back to Top]</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create -n opencity python=3.9.13

conda activate opencity

# Torch (other versions are also ok)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Clone our OpenCity or download it
git clone https://github.com/HKUDS/OpenCity.git
cd OpenCity-main

# Install required libraries
pip install -r requirements.txt
```

<span id='Training OpenCity'/>

### 3. Training OpenCity <a href='#all_catelogue'>[Back to Top]</a>

<span id='Preparing Pre-trained Data'/>

#### 3.1. Preparing Pre-trained Data <a href='#all_catelogue'>[Back to Top]</a>

* The model's generalization capabilities and predictive performance were extensively evaluated using a diverse set of large-scale, real-world public datasets covering various traffic-related data categories, including **Traffic Flow**, **Taxi Demand**, **Bicycle Trajectories**, **Traffic Speed Statistics**, and **Traffic Index Statistics**, from regions across the United States and China, such as New York City, Chicago, Los Angeles, the Bay Area, Shanghai, Shenzhen, and Chengdu. <br />
* These data are organized in [OpenCity-dataset](https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main). Please download it and put it at ./data. Subsequently, unzip all files and run [generate_ca_data.py](https://github.com/HKUDS/OpenCity/blob/main/data/generate_ca_data.py).

<span id='Pre-training'/>

#### 3.2. Pre-training <a href='#all_catelogue'>[Back to Top]</a>

* To pretrain the OpenCity model with different configurations, you can execute the Run.py code. There are some examples:
```
# OpenCity-plus
python Run.py -mode pretrain -model OpenCity -save_pretrain_path OpenCity-plus2.0.pth -batch_size 4 --embed_dim 512 --skip_dim 512 --enc_depth 6

# OpenCity-base
python Run.py -mode pretrain -model OpenCity -save_pretrain_path OpenCity-base2.0.pth -batch_size 8 --embed_dim 256 --skip_dim 256 --enc_depth 3

# OpenCity-mini
python Run.py -mode pretrain -model OpenCity -save_pretrain_path OpenCity-mini2.0.pth -batch_size 16 --embed_dim 128 --skip_dim 128 --enc_depth 3

```

* Parameter setting instructions. The parameter settings consist of two parts: the pretrain config and other configs. To avoid any confusion arising from potential overlapping parameter names, we employ a hyphen (-) to specify the parameters of pretrain config and use a double hyphen (--) to specify the parameters of other configs. Please note that if two parameters have the same name, **the settings of the latter can override those of the former.**

<span id='Evaluating'/>

### 4. Evaluating <a href='#all_catelogue'>[Back to Top]</a>

* **Preparing Checkpoints of OpenCity**. You can download our model using the following link: [OpenCity-Plus](https://huggingface.co/hkuds/OpenCity-Plus/tree/main), [OpenCity-Base](https://huggingface.co/hkuds/OpenCity-Base/tree/main), [OpenCity-Mini](https://huggingface.co/hkuds/OpenCity-Mini/tree/main)

* **Running Evaluation of OpenCity**. You can use our release model weights to evaluate, There is an example as below: 
```
# Use OpenCity-plus to evaluate, please use only one dataset to test (e.g. dataset_use = ['PEMS07M'] in pretrain.config).
python Run.py -mode test -model OpenCity -load_pretrain_path OpenCity-plus.pth -batch_size 2 --embed_dim 512 --skip_dim 512 --enc_depth 6
```

* **Running Evaluation of other baselines**. You can Replace the model name or use ori mode to train and test. For example: 

```
# Run STGCN in ori mode
python Run.py -mode ori -model STGCN -batch_size 64 --real_value False
```

<!--
## Contact
For any questions or feedback, feel free to contact [Zhonghang Li](mailto:bjdwh.zzh@gmail.com).
-->

## Citation

If you find OpenCity useful in your research or applications, please kindly cite:

```
@misc{li2024opencity,
      title={OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction}, 
      author={Zhonghang Li and Long Xia and Lei Shi and Yong Xu and Dawei Yin and Chao Huang},
      year={2024},
      eprint={2408.10269},
      archivePrefix={arXiv}
}
```


<!--
## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat). We also partially draw inspirations from [GraphGPT](https://github.com/HKUDS/GraphGPT). The design of our website and README.md was inspired by [NExT-GPT](https://next-gpt.github.io/), and the design of our system deployment was inspired by [gradio](https://www.gradio.app) and [Baize](https://huggingface.co/spaces/project-baize/chat-with-baize). Thanks for their wonderful works.
-->
