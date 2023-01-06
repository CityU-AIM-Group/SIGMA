## Dataset Preparation

Step 1: Prepare required benchmark datasets. Almost all popular DAOD benchmarks are supported in this project.

```
[DATASET_PATH]
Cityscapes
   └─ cocoAnnotations
   └─ leftImg8bit
      └─ train
      └─ val
   └─ leftImg8bit_foggy
      └─ train
      └─ val
KITTI
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
Sim10k
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
BDD100k
   └─ cocoAnnotations
   └─ images
      └─ train
      └─ val
style-transferred
   └─ VOC2007_to_clipart
   └─ VOC2012_to_clipart
   └─ VOC2007_to_watercolor
   └─ VOC2012_to_watercolor
   └─ VOC2007_to_comic
   └─ VOC2012_to_clipart
clipart
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
watercolor
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
comic
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
VOCdevkit # only used for source-only training
   └─ VOC2007
   └─ VOC2012

```

We follow [EPM](https://github.com/chengchunhsu/EveryPixelMatters) for city-based settings. Pure annotation files are available at [onedrive](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/Eq7hy8iTGBFGpSz19mlSUN0BhIf9dL_oAdONwmPCAn-BRg?e=n5aNyU).
Some datasets are avaliable at [datasets](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/EtqmL4sghYhNn6d9hMiEzJIBeUJpdd0iIUNVJPVjTAixBA?e=t3qCh1).

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

**Cityscapes -> BDD100k** (7-class evaluation w/o class train)
  - You can use the uploaded version from this link [BDD100K](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/EtqmL4sghYhNn6d9hMiEzJIBeUJpdd0iIUNVJPVjTAixBA?e=5D5eiy), which correct the inconsistent class names and remove unused images. 
  - The official website: [BDD100K](https://bdd-data.berkeley.edu/). 

**Pascal VOC 07/12 -> Clipart** 
   - Download the style-transferred Pascal VOC 07/12 datasets from this [link](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/EtqmL4sghYhNn6d9hMiEzJIBeUJpdd0iIUNVJPVjTAixBA?e=5D5eiy), which are borrowed from [D-adapt](https://github.com/thuml/Transfer-Learning-Library/tree/dev-tllib/examples/domain_adaptation/object_detection). 
   - Extract the training set and move them to `style-transferred` directory.
   - Download the clipart dataset from this link [clipart](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/EtqmL4sghYhNn6d9hMiEzJIBeUJpdd0iIUNVJPVjTAixBA?e=5D5eiy), and extract it to `clipart` directory.

**Pascal VOC 07/12 -> Watercolor/Comic** 
   - Download the style-transferred Pascal VOC 07/12 datasets from this [link](https://portland-my.sharepoint.com/:f:/g/personal/wuyangli2-c_my_cityu_edu_hk/EtqmL4sghYhNn6d9hMiEzJIBeUJpdd0iIUNVJPVjTAixBA?e=5D5eiy), which are borrowed from [D-adapt](https://github.com/thuml/Transfer-Learning-Library/tree/dev-tllib/examples/domain_adaptation/object_detection). 
   - Extract the training set and move them to `style-transferred/` directory.
   - Since most images are not used for DAOD tasks, we upload a formatted vesion at this link [](), which removes the unused images to save the disk space.
   - The offical watercolor/comic dataset is available at this link [watercolor/comic](https://github.com/naoto0804/cross-domain-detection), and extract it to ` watercolor/` and `comic/`  directory, respectively.

Step 2: change the data root for your dataset at [paths_catalog.py](../fcos_core/config/paths_catalog.py).

```
DATA_DIR = [$Your dataset root]
```