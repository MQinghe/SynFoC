from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import copy
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage.interpolation import zoom

class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../data/Fundus',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None,
                 img_size = 256, 
                 is_RGB = False
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'DGS', 2:'RIM', 3:'REF', 4:'REF_val'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.img_size = img_size
        self.is_RGB = is_RGB

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(i), phase, 'ROIs/image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()

            if (self.splitid == i or self.splitid == -1) and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                # self.image_list.append({'image': image_path, 'label': gt_path, 'domain_code': i})
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(_img_name)
            print(f'-----Number of domain {i} images: {len(imagelist)}, Excluded: {len(excluded_idxs)}')

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        # self._read_img_into_memory()
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).convert('RGB').resize((self.img_size, self.img_size), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((self.img_size, self.img_size), Image.NEAREST)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index]).convert('RGB').resize((self.img_size, self.img_size), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_DGS:
                Flag = 'DGS'
            elif basename[0] in self.flags_REF:
                Flag = 'REF'
            elif basename[0] in self.flags_RIM:
                Flag = 'RIM'
            elif basename[0] in self.flags_REF_val:
                Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            self.image_pool.append(
                Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
            _target = Image.open(self.image_list[index]['label'])

            if _target.mode is 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)
            self.img_domain_code_pool.append(self.image_list[index]['domain_code'])



    def __str__(self):
        return 'Fundus(phase=' + self.phase+str(self.splitid) + ')'

class ProstateSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../data/ProstateSlice',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4,5,6],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None,
                 img_size = 384,
                 is_RGB = False
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.img_size = img_size
        self.is_RGB = is_RGB

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).resize((self.img_size, self.img_size), Image.LANCZOS)
            _target = Image.open(self.label_pool[index]).resize((self.img_size, self.img_size), Image.NEAREST)
            if _img.mode is 'RGB':
                print('img rgb')
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                print('target rgb')
                _target = _target.convert('L')
            if self.is_RGB:
                _img = _img.convert('RGB')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index]).resize((self.img_size, self.img_size), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            if self.is_RGB:
                _img = _img.convert('RGB')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample




    def __str__(self):
        return 'Prostate(phase=' + self.phase+str(self.splitid) + ')'

class MNMSSegmentation(Dataset):
    """
    MNMS segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../data/mnms',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None,
                 img_size = 288,
                 is_RGB = False
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'vendorA', 2:'vendorB', 3:'vendorC', 4:'vendorD'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.img_size = img_size
        self.is_RGB = is_RGB

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).resize((self.img_size, self.img_size), Image.BILINEAR)
            _target = Image.open(self.label_pool[index]).resize((self.img_size, self.img_size), Image.NEAREST)
            if _img.mode is 'RGB':
                print('img rgb')
                _img = _img.convert('L')
            target_np = np.array(_target)
            new_target = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            for n in range(3):
                new_target[target_np[:, :, n] == 255] = n + 1
            _target = Image.fromarray(new_target)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index]).resize((self.img_size, self.img_size), Image.BILINEAR)
            _target = Image.open(self.label_pool[index]).resize((288,288), Image.NEAREST)
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            # if _target.mode is 'RGB':
            #     _target = _target.convert('L')
            target_np = np.array(_target)
            new_target = np.zeros((288,288), dtype=np.uint8)
            for n in range(3):
                new_target[target_np[:, :, n] == 255] = n + 1
            _target = Image.fromarray(new_target)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample




    def __str__(self):
        return 'MNMS(phase=' + self.phase+str(self.splitid) + ')'

class ACDCSegmentation(Dataset):
    def __init__(self, base_dir=None, split='train', 
                 index=None, weak_transform=None, 
                 strong_transform=None, normal_toTensor=None,
                 img_size = 256,):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.normal_toTensor = normal_toTensor
        self.img_size = img_size
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        elif self.split == 'test':
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if index is not None and self.split == "train":
            self.sample_list = [self.sample_list[i] for i in index]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            image = zoom(image*255, (self.img_size / image.shape[-2], self.img_size / image.shape[-1]), order=0)
            label = zoom(label, (self.img_size / label.shape[-2], self.img_size / label.shape[-1]), order=0)
            image = Image.fromarray(image.astype(np.float32)).convert('L')
            label = Image.fromarray(label.astype(np.uint8)).convert('L')
            sample = {'image': image, 'label': label}
            if self.weak_transform is not None:
                sample = self.weak_transform(sample)
            if self.strong_transform is not None:
                sample['strong_aug'] = self.strong_transform(sample['image'])
            sample = self.normal_toTensor(sample)
        return sample

class BUSISegmentation(Dataset):
    def __init__(self, base_dir=None, phase='train', splitid=1, domain=[1,2],
                 weak_transform=None, strong_tranform=None, normal_toTensor=None,
                 selected_idxs = None,
                 img_size = 256,
                 is_RGB = False):
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'benign', 2:'malignant'}
        self.sample_list = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.img_size = img_size
        self.is_RGB = is_RGB
        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i]+'/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            domain_data_list = []
            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            for image in imagelist:
                if 'mask' not in image:
                    domain_data_list.append([image])
                else:
                    domain_data_list[-1].append(image)
            test_benign_num = int(len(domain_data_list)*0.2)
            train_benign_num = len(domain_data_list) - test_benign_num
            if self.phase == 'test':
                domain_data_list = domain_data_list[-test_benign_num:]
            elif self.phase == 'train':
                domain_data_list = domain_data_list[:train_benign_num]
            else:
                raise Exception('Unknown split...')
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(domain_data_list)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                domain_data_list.pop(exclude_id)
                
            for image_path in domain_data_list:
                self.sample_list.append(image_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path[0].split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.sample_list), excluded_num))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        _img = Image.open(self.sample_list[idx][0]).convert('L').resize((self.img_size, self.img_size), Image.LANCZOS)
        if len(self.sample_list[idx]) == 2:
            if self.phase == 'train':
                _target = Image.open(self.sample_list[idx][1]).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
            else:
                _target = Image.open(self.sample_list[idx][1]).convert('L').resize((256, 256), Image.NEAREST)
        else:
            target_list = []
            for target_path in self.sample_list[idx][1:]:
                target = Image.open(target_path).convert('L')
                target_list.append(np.array(target))
            height, width = target_list[0].shape
            combined_target = np.zeros((height, width), dtype=np.uint8)
            for target in target_list:
                combined_target = np.maximum(combined_target, target)
            if self.phase == 'train':
                _target = Image.fromarray(combined_target).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
            else:
                _target = Image.fromarray(combined_target).convert('L').resize((256, 256), Image.NEAREST)
        
        sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[idx], 'dc': self.img_domain_code_pool[idx]}
        if self.phase == "train":
            if self.weak_transform is not None:
                sample = self.weak_transform(sample)
            if self.strong_transform is not None:
                sample['strong_aug'] = self.strong_transform(sample['image'])
            sample = self.normal_toTensor(sample)
        else:
            sample = self.normal_toTensor(sample)
        return sample


import SimpleITK as sitk
class MSCMRSegSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../data/MS-CMRSeg',
                 phase='train',
                 splitid=2,
                 domain=[1,2],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None,
                 img_size = 192,
                 is_RGB = False
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.phase = phase
        self.domain_name = {1:'C0', 2:'LGE'}
        self.img_size = img_size
        self.is_RGB = is_RGB

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        
        imgs = np.zeros((1,192,192))
        labs = np.zeros((1,192,192))
        self.info = []
        sum_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase)

            imagelist = glob(self._image_dir + '/'+f'*{self.domain_name[i]}.nii*')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir + '/'+f'*{self.domain_name[i]}.nii*'))
            imagelist.sort()
            sum_num += len(imagelist)
            
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
            
            for img_num in range(len(imagelist)):
                itkimg = sitk.ReadImage(imagelist[img_num])
                npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

                imgs = np.concatenate((imgs,npimg),axis=0)

                labname = imagelist[img_num].replace('.nii','_manual.nii')
                itklab = sitk.ReadImage(labname)
                nplab = sitk.GetArrayFromImage(itklab)
                nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

                labs = np.concatenate((labs, nplab), axis=0)

        self.imgs = imgs[1:,:,:]
        self.labs = labs[1:,:,:]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)
            

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, sum_num, excluded_num))

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        npimg = self.imgs[index,:,:]
        nplab = self.labs[index,:,:]
        _img = Image.fromarray(npimg*255).convert('L').resize((self.img_size, self.img_size), Image.LANCZOS)
        if self.phase != 'test':
            _target = Image.fromarray(nplab).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
            if _img.mode is 'RGB':
                print('img rgb')
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                print('target rgb')
                _target = _target.convert('L')
            if self.is_RGB:
                _img = _img.convert('RGB')
            anco_sample = {'image': _img, 'label': _target}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)
            if self.strong_transform is not None:
                anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _target = Image.fromarray(nplab).convert('L').resize((192, 192), Image.NEAREST)
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            if self.is_RGB:
                _img = _img.convert('RGB')
            anco_sample = {'image': _img, 'label': _target}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample




    def __str__(self):
        return 'Prostate(phase=' + self.phase+str(self.splitid) + ')'

if __name__ == '__main__':
    import custom_transforms as tr
    from utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        # tr.RandomHorizontalFlip(),
        # tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    voc_train = FundusSegmentation(#split='train1',
    splitid=[1,2],lb_domain=2,lb_ratio=0.2,
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)
    print(len(dataloader))
    for ii, sample in enumerate(dataloader):
        # print(sample)
        exit(0)
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = tmp
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

            break
    plt.show(block=True)


