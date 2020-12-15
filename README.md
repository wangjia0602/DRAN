# DRAN

This repository is an official PyTorch implementation of the paper **"Deep Learning Electron Microscopy Based on Deep Residual Attention Network
"**.
If you find our work useful in your research or publication, please cite our work。
We provide the source code and the corresponding dataset, and then increase the pre-trained model depending on the situation.

## Dependencies
* Python 3.6
* PyTorch = 1.4.0

## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/wangjia0602/DRAN
cd DRAN
```

## Dataset
Our own microscope dataset

[Butterfly Dataset](https://doi.org/10.5281/zenodo.4320132)

You can train and test your models with the datasets:

[DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

You can evaluate your models with widely-used benchmark datasets:

[Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),

[Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests),

[B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),

[Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).

You can use the data we provide for training and testing, or you can make your own dataset.

## How to train and test

```DRAN.py``` the model structure
```dataset_plus.py``` dataset preprocessing and data type conversion
```srnn.py``` the main implementation of training and testing
```test.py``` Parameter management during training
```train.py``` Parameter management during testing
```utility.py   utils.py``` Some instrumental functions used in the training process

You can test our super-resolution algorithm with your own images. Place your images in any folder you want and (like ``test/<your_image>``) We support **png** files.

Download the required data set, and then change the training directory and other parameters in the train and test files.
```bash
python3.6 train.py --cuda       # You are now in */DRAN, --cuda is to use gpu for training
```
If you want to test
```bash
python3.6 test.py --cuda       # You are now in */DRAN, --cuda is to use gpu for training
```
