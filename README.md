# PETCT-RMA-Detection
This repository contains machine learning models to detect repiratory motion misalingment artifact (RMA) in PET CT images.
The pretrained random forest model needs PET and CT segmentations for four organs of liver, spleen, lung, and heart to decide about the presence of RMA. 
The PET and CT segmentation models can be found in [Organ-Segmentation](https://github.com/YazdanSalimi/Organ-Segmentation) repository. please download the trained random forest models [here](https://drive.google.com/drive/folders/1EFIRENGMTF-e5k6lOtL-8pQayFXn60Z7?usp=sharing).
The inference example is presented in "inference-example.py" file. 
## Installation
`pip install git+https://github.com/YazdanSalimi/PETCT-RMA-Detection.git`



