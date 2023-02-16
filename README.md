# Visual SLAM with Vision Transformers(ViT)
To function in uncharted areas, intelligent mobile robots need simultaneous localization and mapping (SLAM). Nevertheless, standard feature extraction algorithms that traditional visual SLAM systems rely on have trouble dealing with texture-less regions and other complicated scenes, which limits the development of visual SLAM. Deep learning-based feature point extraction research demonstrate that this method outperforms standard methods in dealing with complicated scenarios.
This repository presents a Visual SLAM algorithm for mobile robot based on Vision Transformers.
The Project is split into 2 phases:
* Developing a Vision Transformer Based Feature Extraction Model and Object Detection Model for RGB-D and RGB Datasets
* Incoporating the Vision Trasnformer Based Model with Classical SLAM algorithms like EKF(Extended Kalman Filter)

# Phase One: Vision Transformer Based Model
This section presents an implementation of two models: a SOTA object detection based vision transformer model DETR(Detection Transformer) [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) and a standard vision transformer model.
* detr folder contains the implementation of DETR, pretrained versions can be found here [DETR Facebook](https://github.com/facebookresearch/detr)
* vit-pytorch contains the implementation of standard vision transformer.
* libs contains options for training, testing and custom dataloaders for TUM, NYU, KITTI datasets.
* Dependencies: requirements.txt

DETR Architecture ![DETR](detr/DETR.png)

Standard ViT Architecture ![ViT](vit-pytorch/VIT.png)

PS: This is a work in progress, due to limited compute resource, I am yet to finetune the DETR model and standard vision transformer on TUM RGB-D dataset and run inference. Progressively, I'd do this.



