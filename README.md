# Attention-based-Melanoma-Classifiers
Implementation of four novel attention-based CNN classifiers based on EfficientNet B3 backbone.

Official code for the paper: *Going Smaller: Attention-based Models For Automated Melanoma Diagnosis*

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

in this work, we present four attention-based convolutional neural network (CNN) architectures
based on the EfficientNet-B3 backbone for accurate classification of skin lesions, with a specific focus on melanoma detection.
These novel models are significantly smaller than most existing models, yet they demonstrate comparable performance
to those larger alternatives when tested on an independent dataset. Our best model achieved a ROC-AUC of 0.82, an
F1-score of 0.84, PR-AUC of 0.54, a sensitivity of 0.96 (with a predefined threshold of 95%), and a specificity of 0.21 on the
melanoma detection task. Comparative analysis with the top three winners of the International Skin Imaging Collaboration
(ISIC) 2020 challenge on the test set demonstrated superior performance, outperforming the second and third winners and
achieving comparable results to the first winner while using up to 95% fewer parameters.

## Installation

Install all dependencies from requirements.txt. Download the train and test datasets and place them in the data folder along with their corresponding .csv metadata files.

## Usage
To run inference, use the ```predict.py``` script. If you do not wish to connect your script to Wandb, remove the wandb.log commands. Otherwise, insert your Wandb credentials before running.
``` python
usage: predict_custom.py [-h] --kernel-type KERNEL_TYPE [--data-dir DATA_DIR] --image-size IMAGE_SIZE --enet-type ENET_TYPE
                         [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--out-dim OUT_DIM] [--use-amp] [--use-meta] [--DEBUG]
                         [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--sub-dir SUB_DIR] [--eval {best,final,epoch10,epoch20,epoch30}]
                         [--n-test N_TEST] [--CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES] [--n-meta-dim N_META_DIM] [--hard_samples]
                         [--fold FOLD]

optional arguments:
  -h, --help            show this help message and exit
  --kernel-type KERNEL_TYPE
  --data-dir DATA_DIR
  --image-size IMAGE_SIZE
  --enet-type ENET_TYPE
  --batch-size BATCH_SIZE
  --num-workers NUM_WORKERS
  --out-dim OUT_DIM
  --use-amp
  --use-meta
  --DEBUG
  --model-dir MODEL_DIR
  --log-dir LOG_DIR
  --sub-dir SUB_DIR
  --eval {best,final,epoch10,epoch20,epoch30}
  --n-test N_TEST
  --CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  --n-meta-dim N_META_DIM
  --hard_samples
  --fold FOLD
```


Below is an example code snippet to demonstrate how to run inference using our project with SGE-B3 model:

```python
python predict.py --kernel-type 9c_SGE-b3_384_384_35ep  --data-folder 512 --image-size 384 --enet-type SGE-B3 --fold 0,1,2,3,4 --model-dir /weights/

