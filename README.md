# [SIGMA: Semantic-complete Graph Matching For Domain Adaptive Object Detection (CVPR-22 ORAL)](https://arxiv.org/pdf/2203.06398.pdf)

[[Arxiv](https://arxiv.org/pdf/2203.06398.pdf)] [[知乎](https://zhuanlan.zhihu.com/p/492956292)]

By [Wuyang Li](https://wymancv.github.io/wuyang.github.io/)

### 🎉News! If you feel that your DAOD research has hit a bottleneck, welcome to check out our latest work on [Adaptive Open-set Object Detection](https://github.com/CityU-AIM-Group/SOMA), which extends the target domain to the open set! 



Three branches of the project:
- Main branch (SIGMA): ```git clone https://github.com/CityU-AIM-Group/SIGMA.git```
- [SIGMA++](https://github.com/CityU-AIM-Group/SIGMA/tree/SIGMA++) branch: ```git clone -b SIGMA++ https://github.com/CityU-AIM-Group/SIGMA.git```
- [FRCNN-SIGMA++](https://github.com/CityU-AIM-Group/SIGMA/tree/SIGMA++) branch: ```git clone -b FRCNN-SIGMA++ https://github.com/CityU-AIM-Group/SIGMA.git```

The features of [SIGMA++](https://github.com/CityU-AIM-Group/SIGMA/tree/SIGMA++):
- More datasets and instructions
- More stable and better results
- From graph to hypergraph

![image](./assets/matching_visualization.png)

 SIGMA++ has found its final home now, indicating the end of this series of works. The growth of SIGMA++ is full of frustration: 👶 ➡  🧒. 
 
 [SCAN](https://ojs.aaai.org/index.php/AAAI/article/view/20031) ➡ [SCAN++](https://ieeexplore.ieee.org/document/9931144/) ➡ [SIGMA](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_SIGMA_Semantic-Complete_Graph_Matching_for_Domain_Adaptive_Object_Detection_CVPR_2022_paper.pdf) ➡ [SIGMA++](./assets/manuscript.pdf)

The main idea of the series of works: *Model fine-grained feature points with graphs.* We 
 sincerely appreciate for all the readers showing interest in our works. 

Honestly, due to the limited personal ability, our works still have many limitations, e.g., sub-optimal and redundant designs. Please forgive me. Nevertheless, we hope our works can inspire lots of good idea.

Best regards,\
[Wuyang Li](https://wymancv.github.io/wuyang.github.io/)\
E-mail: wuyangli2-c@my.cityu.edu.hk 


## 💡 Preparation
#### Installation

Check [INSTALL.md](./docs/INSTALL.md) for installation instructions. If you have any problem, feel free to screenshot your issue for me. Thanks.

#### Data preparation

More detailed preparation instructions are available [here](./docs/DATASETS.md).

Step 1: Prepare benchmark datasets. 

We follow [EPM](https://github.com/chengchunhsu/EveryPixelMatters) to construct the training and testing set by three following settings. Annotation files are available at [onedrive](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/Eq7hy8iTGBFGpSz19mlSUN0BhIf9dL_oAdONwmPCAn-BRg?e=n5aNyU).

**Cityscapes -> Foggy Cityscapes**
  - Download Cityscapes and Foggy Cityscapes dataset from the [link](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *leftImg8bit_trainvaltest.zip* for Cityscapes and *leftImg8bit_trainvaltest_foggy.zip* for Foggy Cityscapes.
  - Download and extract the converted annotation from the following links: [Cityscapes and Foggy Cityscapes (COCO format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EZRXq3_5R_RKpAuTwjyYpWYBTjgKWZNuEjsgoYky31a96g?e=hfWAyl)
<!--   - (https://drive.google.com/file/d/1LRNXW2Wee8tjuxc5gjVsFQv49vA_SBtk/view?usp=sharing). -->
  - Extract the training sets from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` to `Cityscapes/leftImg8bit/` directory.
  - Extract the training and validation set from *leftImg8bit_trainvaltest_foggy.zip*, then move the folder `leftImg8bit_foggy/train/` and `leftImg8bit_foggy/val/` to `Cityscapes/leftImg8bit_foggy/` directory.
  
 **Sim10k -> Cityscapes** (class car only)
  - Download Sim10k dataset and Cityscapes dataset from the following links: [Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *repro_10k_images.tgz* and *repro_10k_annotations.tgz* for Sim10k and *leftImg8bit_trainvaltest.zip* for Cityscapes.
  - Download and extract the converted annotation from the following links: [Sim10k (VOC format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EQt48_9D1XtIiVE9GK3hFIYBQNOVSW4OfdZPtQAcCkS7bw?e=8NCweC) and [Cityscapes (COCO format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EZRXq3_5R_RKpAuTwjyYpWYBTjgKWZNuEjsgoYky31a96g?e=hfWAyl)
<!--   - (https://drive.google.com/file/d/1LRNXW2Wee8tjuxc5gjVsFQv49vA_SBtk/view?usp=sharing). -->
  - Extract the training set from *repro_10k_images.tgz* and *repro_10k_annotations.tgz*, then move all images under `VOC2012/JPEGImages/` to `Sim10k/JPEGImages/` directory and move all annotations under `VOC2012/Annotations/` to `Sim10k/Annotations/`.
  - Extract the training and validation set from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` and `leftImg8bit/val/` to `Cityscapes/leftImg8bit/` directory.
  
**KITTI -> Cityscapes** (class car only)
  - Download KITTI dataset and Cityscapes dataset from the following links: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *data_object_image_2.zip* for KITTI and *leftImg8bit_trainvaltest.zip* for Cityscapes.
  - Download and extract the converted annotation from the following links: [KITTI (VOC format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EWqP3z9BpVNLlG3a_qGBO1EBNO7XO4GGaDlipixnlgc7rQ?e=LPBV5j) and [Cityscapes (COCO format)](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EZRXq3_5R_RKpAuTwjyYpWYBTjgKWZNuEjsgoYky31a96g?e=hfWAyl).
  - Extract the training set from *data_object_image_2.zip*, then move all images under `training/image_2/` to `KITTI/JPEGImages/` directory.
  - Extract the training and validation set from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` and `leftImg8bit/val/` to `Cityscapes/leftImg8bit/` directory.

```
[DATASET_PATH]
└─ Cityscapes
   └─ cocoAnnotations
   └─ leftImg8bit
      └─ train
      └─ val
   └─ leftImg8bit_foggy
      └─ train
      └─ val
└─ KITTI
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
└─ Sim10k
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
```

Step 2: change the data root for your dataset at [paths_catalog.py](https://github.com/CityU-AIM-Group/SIGMA/blob/main/fcos_core/config/paths_catalog.py).

```
DATA_DIR = [$Your dataset root]
```

## 📦 Well-trained models

The ImageNet pretrained VGG-16 backbone (w/o BN) is available at [link](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/ESOgJbvystdDiGbMLiGnL50BvxxwSJ3LjR22yxo9-OdTOA?e=5cA2xY). You can use it if you cannot download the model through the link in the config file. \
The well-trained models are available at this [link](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/Eh94jXa1NSxAilUAE68-T0MBckTxK3Tm-ggmzZRJTHNHww?e=B30DNw)).

1) We can get higher results than the reported ones with tailor-tuned hyperparameters.
2) E2E indicates end-to-end training for better reproducibility. Our config files are used for end-to-end training.
3) Two-stage/ longer training and turning learning rate will make the results more stable and get higer mAP/AP75.
4) After correcting a default hyper-parameter (as explained in the config file), Sim10k to City achieves better results than the reported ones.
5) You can set MODEL.MIDDLE_HEAD.GM.WITH_CLUSTER_UPDATE False to accelerate training greatly with ignorable performance drops. You'd better also make this change for bs=2 since we found it more friendly for the small batch-size training.
6) Results will be stable after the learning rate decline (in the training schedule).

| Source| Target| E2E|Metric | Backbone |   mAP	 | AP@50 |  AP@75 |	 file |		
| :-----:|:----:|:----: | :----: | :----:| :----: |:-----:| :----: | :----: | 
| City 	|Foggy 	| |COCO |V-16|24.0 |43.6|23.8|city_to_foggy_vgg16_43.58_mAP.pth|
| City 	|Foggy  | |COCO |V-16| 24.3 |43.9|22.6| city_to_foggy_vgg16_43.90_mAP.pth|
| City 	|Foggy  |$\checkmark$ |COCO |V-16| 22.0 |43.5|21.8| reproduced|
| City 	|Foggy  | |COCO |R-50| 22.7 |44.3|21.2| city_to_foggy_res50_44.26_mAP.pth|
| City  | BDD100k| |COCO|V-16 | - |32.7|- |city_to_bdd100k_vgg16_32.65_mAP.pth|
| Sim10k| City | |COCO|V-16 | 33.4 |57.1 |33.8 |sim10k_to_city_vgg16_53.73_mAP.pth|
| Sim10k 	|City  |$\checkmark$ |COCO |V-16| 32.1 |55.2|32.1| reproduced|
| KITTI | City | |COCO|V-16 | 22.6 |46.6 |20.0 |kitti_to_city_vgg16_46.45_mAP.pth|

## 🔥 Get start
NOTE: In the code comments, there is a small correction about batchsize: IMS_PER_BATACH=4 indicates 4 images per domain.

Train the model from the scratch with the default setting (batchsize = 4):
```
python tools/train_net_da.py \
        --config-file configs/SIGMA/xxx.yaml \

```
Test the well-trained model:
```
python tools/test_net.py \
        --config-file configs/SIGMA/xxx.yaml \
        MODEL.WEIGHT well_trained_models/xxx.pth
```
For example: test cityscapes to foggy cityscapes with VGG16 backbone.
```
python tools/test_net.py \
         --config-file configs/SIGMA/sigma_vgg16_cityscapace_to_foggy.yaml \
         MODEL.WEIGHT well_trained_models/city_to_foggy_vgg16_43.58_mAP.pth

```

## ✨ Quick Tutorials
1) See [sigma_vgg16_cityscapace_to_foggy.yaml](https://github.com/CityU-AIM-Group/SIGMA/blob/main/configs/SIGMA/sigma_vgg16_cityscapace_to_foggy.yaml) to understand APIs.
2) We modify the [trainer](https://github.com/CityU-AIM-Group/SIGMA/blob/main/fcos_core/engine/trainer.py) to meet the requirements of SIGMA.
3) GM is integrated in this "middle layer": [graph_matching_head](https://github.com/CityU-AIM-Group/SIGMA/blob/main/fcos_core/modeling/rpn/fcos/graph_matching_head.py).
4) Node sampling is conducted together with fcos loss: [loss](https://github.com/CityU-AIM-Group/SIGMA/blob/main/fcos_core/modeling/rpn/fcos/loss.py).



## 📝 Citation 
If you think this work is helpful for your project, please give it a star and citation. We sincerely appreciate for your acknowledgments.

```BibTeX
@inproceedings{li2022sigma,
  title={SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection},
  author={Li, Wuyang and Liu, Xinyu and Yuan, Yixuan},
  booktitle={CVPR},
  year={2022}
}
```
Relevant project:
```BibTeX
@inproceedings{li2022scan,
  title={SCAN: Cross Domain Object Detection with Semantic Conditioned Adaptation},
  author={Li, Wuyang and Liu, Xinyu and Yao, Xiwen and Yuan, Yixuan},
  booktitle={AAAI},
  year={2022}
}
```

## 🤞 Acknowledgements 

We mainly appreciate for these good projects and their authors' hard-working.
- This work is based on [EPM](https://github.com/chengchunhsu/EveryPixelMatters). 
- The implementation of our anchor-free detector is from [FCOS](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f), which highly relies on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
- The style-transferred data is from [D_adapt](https://github.com/thuml/Transfer-Learning-Library/tree/dev-tllib/examples/domain_adaptation/object_detection). 
- The faster-rcnn-based implementation is based on [DA-FRCNN](https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch). 

## 📒 Abstract

Domain Adaptive Object Detection (DAOD) leverages a labeled source domain to learn an object detector generalizing to a novel target domain free of annotations. Recent advances align class-conditional distributions through narrowing down cross-domain prototypes (class centers). Though great success, these works ignore the significant within-class variance and the domain-mismatched semantics within the training batch, leading to a sub-optimal adaptation. To overcome these challenges, we propose a novel SemantIc-complete Graph MAtching (SIGMA) framework for DAOD, which completes mismatched semantics and reformulates the adaptation with graph matching. Specifically, we design a Graph-embedded Semantic Completion module (GSC) that completes mismatched semantics through generating hallucination graph nodes in missing categories. Then, we establish cross-image graphs to model class-conditional distributions and learn a graph-guided memory bank for better semantic completion in turn. After representing the source and target data as graphs, we reformulate the adaptation as a graph matching problem, i.e., finding well-matched node pairs across graphs to reduce the domain gap, which is solved with a novel Bipartite Graph Matching adaptor (BGM). In a nutshell, we utilize graph nodes to establish semantic-aware node affinity and leverage graph edges as quadratic constraints in a structure-aware matching loss, achieving fine-grained adaptation with a node-to-node graph matching. Extensive experiments demonstrate that our method outperforms existing works significantly.

![image](./assets/overall.png)
