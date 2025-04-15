# SynFoC
### 1. Introduction

This repository contains the implementation of the paper **[Steady Progress Beats Stagnation: Mutual Aid of Foundation and Conventional Models in Mixed Domain Semi-Supervised Medical Image Segmentation](https://github.com/MQinghe/SynFoC)**
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2025*

### 2. Dataset Construction

The dataset needs to be divided into two folders for training and testing. The training and testing data should be in the format of the "data_format" folder.

### 3. Train

`code/train.py` is the implementation of our method .

Modify the paths in lines 770 to 817 of the code.

```python
if args.dataset == 'fundus':
    train_data_path='../../data/Fundus' # the folder of fundus dataset
elif args.dataset == 'prostate':
    train_data_path="../../data/ProstateSlice" # the folder of prostate dataset
elif args.dataset == 'MNMS':
    train_data_path="../../data/mnms" # the folder of M&Ms dataset
elif args.dataset == 'BUSI':
    train_data_path="../../data/Dataset_BUSI_with_GT" # the folder of BUSI dataset
```

then simply run:

```python
python train.py --dataset ... --lb_domain ... --lb_num ... --save_name ... --gpu 0 --AdamW --warmup --model MedSAM
```

### 4. Test

To run the evaluation code, please update the path of the dataset in `test.py`:

Modify the paths in lines 248 to 283 of the code.

then simply run:

```
python test.py --dataset ... --save_name ... --gpu 0
```

### 5. Acknowledgement

This project is based on the code from the [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [SAMed](https://github.com/hitachinsk/SAMed) project.

Thanks a lot for their great works.
