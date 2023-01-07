## Installation

### Requirements:
- PyTorch >= 1.0. (We use torch==1.4.0) Installation instructions can be found in https://pytorch.org/get-started/locally/.
- torchvision==0.2.1
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9 
- (optional) OpenCV for the webcam demo

### Step-by-step installation
```
conda create --name SIGMA python=3.7
conda activate SIGMA

# this installs the right pip and dependencies for the fresh python
conda install ipython

# FCOS and coco api dependencies
pip install ninja yacs cython matplotlib tqdm

# follow PyTorch installation in https://pytorch.org/get-started/locally/

conda install cudatoolkit=10.1 # 10.0, 10.1, 10.2, 11+ all can work!
pip install torch==1.4.0 # later is ok!
pip install --no-deps torchvision==0.2.1 

export INSTALL_DIR=$PWD

# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/CityU-AIM-Group/SIGMA.git
cd SIGMA

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

# If you meet the python version problem, 
# pls check the scipy verison (we use 1.6.0),
# since the automatically installed 1.8+ version 
# may not support python 3.7.

unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```
