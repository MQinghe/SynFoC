import argparse
import logging
import os
import random
import shutil
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from networks.unet_model import UNet
from networks.wrn import build_WideResNet
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation, BUSISegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from torch.cuda.amp import autocast, GradScaler
import contextlib
import matplotlib.pyplot as plt 

from torch.optim.lr_scheduler import LambdaLR
import math
from medpy.metric import binary
from segment_anything import sam_model_registry
from importlib import import_module
from scipy.ndimage import zoom
import cv2
from itertools import chain
from skimage.measure import label

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'MNMS', 'BUSI'])
parser.add_argument("--save_name", type=str, default="", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--model", type=str, default="MedSAM", help="model_name")
parser.add_argument("--max_iterations", type=int, default=60000, help="maximum epoch number to train")
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

parser.add_argument("--label_bs", type=int, default=4, help="labeled_batch_size per gpu")
parser.add_argument("--unlabel_bs", type=int, default=4)
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=40)
parser.add_argument('--lb_ratio', type=float, default=0)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--save_img',action='store_true')
parser.add_argument('--save_model',action='store_true')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='../checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_Unet_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def update_SAM_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_A_linear, A_linear in zip(ema_model.w_As, model.w_As):
        ema_A_linear.weight.data.mul_(alpha).add_(A_linear.weight.data, alpha=1 - alpha)
    for ema_B_linear, B_linear in zip(ema_model.w_Bs, model.w_Bs):
        ema_B_linear.weight.data.mul_(alpha).add_(B_linear.weight.data, alpha=1 - alpha)
    for ema_param, param in zip(ema_model.sam.mask_decoder.parameters(), model.sam.mask_decoder.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_param, param in zip(ema_model.sam.prompt_encoder.parameters(), model.sam.prompt_encoder.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cycle(iterable: Iterable):
    """Make an iterator returning elements from the iterable.

    .. note::
        **DO NOT** use `itertools.cycle` on `DataLoader(shuffle=True)`.\n
        Because `itertools.cycle` saves a copy of each element, batches are shuffled only at the first epoch. \n
        See https://docs.python.org/3/library/itertools.html#itertools.cycle for more details.
    """
    while True:
        for x in iterable:
            yield x

def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer

@torch.no_grad()
def test(args, model, test_dataloader, epoch, writer, model_name):
    model.eval()
    val_loss = 0.0
    val_dice = [0.0] * n_part
    val_dc, val_jc, val_hd, val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
    domain_num = len(test_dataloader)
    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(2)
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_loss = 0.0
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
        domain_code = i+1
        for batch_num,sample in enumerate(cur_dataloader):
            assert(domain_code == sample['dc'][0].item())
            mask = sample['label']
            if args.dataset == 'fundus':
                lb_mask = (mask<=128) * 2
                lb_mask[mask==0] = 1
                mask = lb_mask
            elif args.dataset == 'prostate':
                mask = mask.eq(0).long()
            elif args.dataset == 'MNMS':
                mask = mask.long()
            elif args.dataset == 'BUSI':
                mask = mask.eq(255).long()
            if model_name == 'SAM':
                data = sample['image'].cuda()
                output = model(data, multimask_output, args.img_size)['masks']
            elif model_name == 'unet':
                data = sample['unet_size_img'].cuda()
                output = model(data)
            pred_label = torch.max(torch.softmax(output,dim=1), dim=1)[1]
            pred_label = torch.from_numpy(zoom(pred_label.cpu(), (1, patch_size / data.shape[-2], patch_size / data.shape[-1]), order=0))
            
            if args.dataset == 'fundus':
                pred_label = to_2d(pred_label)
                mask = to_2d(mask)
                pred_onehot = pred_label.clone()
                mask_onehot = mask.clone()
            elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                pred_onehot = pred_label.clone().unsqueeze(1)
                mask_onehot = mask.clone().unsqueeze(1)
            elif args.dataset == 'MNMS':
                pred_onehot = to_3d(pred_label)
                mask_onehot = to_3d(mask)
            dice = dice_calcu[args.dataset](np.asarray(pred_label.cpu()),mask.cpu())
            
            dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
            for j in range(len(data)):
                for i, p in enumerate(part):
                    dc[i] += binary.dc(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                    jc[i] += binary.jc(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                    if pred_onehot[j,i].float().sum() < 1e-4:
                        hd[i] += 100
                        asd[i] += 100
                    else:
                        hd[i] += binary.hd95(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                        asd[i] += binary.asd(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
            for i, p in enumerate(part):
                dc[i] /= len(data)
                jc[i] /= len(data)
                hd[i] /= len(data)
                asd[i] /= len(data)
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]
                domain_val_dc[i] += dc[i]
                domain_val_jc[i] += jc[i]
                domain_val_hd[i] += hd[i]
                domain_val_asd[i] += asd[i]
        
        domain_val_loss /= len(cur_dataloader)
        val_loss += domain_val_loss
        writer.add_scalar('{}_val/domain{}/loss'.format(model_name, domain_code), domain_val_loss, epoch)
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
            domain_val_dc[i] /= len(cur_dataloader)
            val_dc[i] += domain_val_dc[i]
            domain_val_jc[i] /= len(cur_dataloader)
            val_jc[i] += domain_val_jc[i]
            domain_val_hd[i] /= len(cur_dataloader)
            val_hd[i] += domain_val_hd[i]
            domain_val_asd[i] /= len(cur_dataloader)
            val_asd[i] += domain_val_asd[i]
        for n, p in enumerate(part):
            writer.add_scalar('{}_val/domain{}/val_{}_dice'.format(model_name, domain_code, p), domain_val_dice[n], epoch)
        text = 'domain%d epoch %d : loss : %f' % (domain_code, epoch, domain_val_loss)
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dice: %f, ' % (p, domain_val_dice[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dc: %f, ' % (p, domain_val_dc[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_jc: %f, ' % (p, domain_val_jc[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_hd: %f, ' % (p, domain_val_hd[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_asd: %f, ' % (p, domain_val_asd[n])
        logging.info(text)
        
    model.train()
    val_loss /= domain_num
    writer.add_scalar('{}_val/loss'.format(model_name), val_loss, epoch)
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
        val_dc[i] /= domain_num
        val_jc[i] /= domain_num
        val_hd[i] /= domain_num
        val_asd[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('{}_val/val_{}_dice'.format(model_name, p), val_dice[n], epoch)
    text = 'epoch %d : loss : %f' % (epoch, val_loss)
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_dice: %f, ' % (p, val_dice[n])
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_dc: %f, ' % (p, val_dc[n])
    text += '\t'
    for n, p in enumerate(part):
        text += 'val_%s_jc: %f, ' % (p, val_jc[n])
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_hd: %f, ' % (p, val_hd[n])
    text += '\t'
    for n, p in enumerate(part):
        text += 'val_%s_asd: %f, ' % (p, val_asd[n])
    logging.info(text)
    return val_dice
    
def to_2d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    temp_prob = input_tensor == torch.ones_like(input_tensor)
    tensor_list.append(temp_prob)
    temp_prob2 = input_tensor > torch.zeros_like(input_tensor)
    tensor_list.append(temp_prob2)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def to_3d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(1, 4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

class statistics(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.record = []
        self.num = 0
        self.avg = 0

    def update(self, val):
        self.record.append(val)
        self.num += 1
        self.avg = sum(self.record) / self.num

def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr

    def create_model(model_name=None, ema=False):
        # Network definition
        if model_name == 'SAM':
            logging.info("load from {}".format(args.ckpt))
            sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
            pkg = import_module(args.module)
            model = pkg.LoRA_Sam(sam, args.rank)
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model.cuda(), img_embedding_size
        elif model_name == 'unet':
            model = UNet(n_channels = num_channels, n_classes = num_classes+1)
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model.cuda()
        else:
            raise Exception('Please provide model name.')

    SAM_model, img_embedding_size = create_model(model_name='SAM')
    ema_SAM_model, _ = create_model(model_name='SAM', ema=True)
    unet_model = create_model(model_name='unet')
    ema_unet_model = create_model(model_name='unet', ema=True)
    
    low_res = img_embedding_size * 4

    max_iterations = args.max_iterations
    weak = transforms.Compose([tr.RandomScaleCrop(args.img_size),
            tr.RandomScaleRotate(fillcolor=fillcolor),
            tr.RandomHorizontalFlip(),
            tr.elastic_transform()
            ])
    
    strong = transforms.Compose([
            tr.Brightness(min_v, max_v),
            tr.Contrast(min_v, max_v),
            tr.GaussianBlur(kernel_size=int(0.1 * args.img_size), num_channels=num_channels),
    ])

    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0,1]),
        tr.ToTensor(low_res=low_res, unet_size=patch_size)
    ])

    domain_num = args.domain_num
    domain = list(range(1,domain_num+1))
    if args.dataset == 'fundus':
        domain_len = [50, 99, 320, 320]
    elif args.dataset == 'prostate':
        domain_len = [225, 305, 136, 373, 338, 133]
    elif args.dataset == 'MNMS':
        domain_len = [1030, 1342, 525, 550]
    elif args.dataset == 'BUSI':
        domain_len = [350, 168]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain-1]
    if args.lb_ratio > 0:
        lb_num = int(sum(domain_len) * args.lb_ratio)
    else:
        lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    test_dataset = []
    test_dataloader = []
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor, img_size=args.img_size)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor, img_size=args.img_size)
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor, img_size=args.img_size)
        test_dataset.append(cur_dataset)
    if not args.eval:
        lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
        ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    iter_num = 0
    start_epoch = 0

    # set to train
    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(num_classes+1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        sam_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, SAM_model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        sam_optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    unet_optimizer = optim.SGD(unet_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    logging.info("{} iterations per epoch".format(args.num_eval_iter))

    max_epoch = max_iterations // args.num_eval_iter
    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part
    stu_best_dice = [0.0] * n_part
    stu_best_dice_iter = [-1] *n_part
    stu_best_avg_dice = 0.0
    stu_best_avg_dice_iter = -1
    stu_dice_of_best_avg = [0.0] * n_part

    iter_num = int(iter_num)

    threshold = args.threshold

    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext

    for epoch_num in range(start_epoch, max_epoch):
        SAM_model.train()
        ema_SAM_model.train()
        unet_model.train()
        ema_unet_model.train()
        p_bar = tqdm(range(args.num_eval_iter))
        p_bar.set_description(f'No. {epoch_num+1}')
        self_conf_sta, mutual_conf_sta, ratio_sta, SAM_ulb_dice_sta, unet_ulb_dice_sta, ensemble_ulb_dice_sta = statistics(), statistics(), statistics(), [statistics() for _ in range(n_part)], [statistics() for _ in range(n_part)], [statistics() for _ in range(n_part)]
        for i_batch in range(1, args.num_eval_iter+1):
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_x_w, lb_y = lb_sample['image'], lb_sample['label']
            ulb_x_w, ulb_x_s, ulb_y = ulb_sample['image'], ulb_sample['strong_aug'], ulb_sample['label']
            lb_low_res_y, ulb_low_res_y = lb_sample['low_res_label'], ulb_sample['low_res_label']
            lb_unet_size_x_w, lb_unet_size_y = lb_sample['unet_size_img'], lb_sample['unet_size_label']
            ulb_unet_size_x_w, ulb_unet_size_x_s, ulb_unet_size_y = ulb_sample['unet_size_img'], ulb_sample['unet_size_strong_aug'], ulb_sample['unet_size_label']
            
            if args.dataset == 'fundus':
                lb_mask = (lb_y<=128) * 2
                lb_mask[lb_y==0] = 1
                ulb_mask = (ulb_y<=128) * 2
                ulb_mask[ulb_y==0] = 1
                lb_low_res_mask = (lb_low_res_y<=128) * 2
                lb_low_res_mask[lb_low_res_y==0] = 1
                ulb_low_res_mask = (ulb_low_res_y<=128) * 2
                ulb_low_res_mask[ulb_low_res_y==0] = 1
                lb_unet_size_mask = (lb_unet_size_y<=128) * 2
                lb_unet_size_mask[lb_unet_size_y==0] = 1
                ulb_unet_size_mask = (ulb_unet_size_y<=128) * 2
                ulb_unet_size_mask[ulb_unet_size_y==0] = 1
            elif args.dataset == 'prostate':
                lb_mask = lb_y.eq(0).long()
                ulb_mask = ulb_y.eq(0).long()
                lb_low_res_mask = lb_low_res_y.eq(0).long()
                ulb_low_res_mask = ulb_low_res_y.eq(0).long()
                lb_unet_size_mask = lb_unet_size_y.eq(0).long()
                ulb_unet_size_mask = ulb_unet_size_y.eq(0).long()
            elif args.dataset == 'MNMS':
                lb_mask = lb_y.long()
                ulb_mask = ulb_y.long()
                lb_low_res_mask = lb_low_res_y.long()
                ulb_low_res_mask = ulb_low_res_y.long()
                lb_unet_size_mask = lb_unet_size_y.long()
                ulb_unet_size_mask = ulb_unet_size_y.long()
            elif args.dataset == 'BUSI':
                lb_mask = lb_y.eq(255).long()
                ulb_mask = ulb_y.eq(255).long()
                lb_low_res_mask = lb_low_res_y.eq(255).long()
                ulb_low_res_mask = ulb_low_res_y.eq(255).long()
                lb_unet_size_mask = lb_unet_size_y.eq(255).long()
                ulb_unet_size_mask = ulb_unet_size_y.eq(255).long()
            lb_x_w, lb_mask, ulb_x_w, ulb_x_s, ulb_mask = lb_x_w.cuda(), lb_mask.cuda(), ulb_x_w.cuda(), ulb_x_s.cuda(), ulb_mask.cuda()
            lb_unet_size_x_w, ulb_unet_size_x_w, ulb_unet_size_x_s = lb_unet_size_x_w.cuda(), ulb_unet_size_x_w.cuda(), ulb_unet_size_x_s.cuda()
            lb_low_res_mask, ulb_low_res_mask = lb_low_res_mask.cuda(), ulb_low_res_mask.cuda()
            lb_unet_size_mask, ulb_unet_size_mask = lb_unet_size_mask.cuda(), ulb_unet_size_mask.cuda()

            with amp_cm():
                with torch.no_grad():
                    sam_output_ulb_x_w = ema_SAM_model(ulb_x_w, multimask_output, args.img_size)
                    sam_logits_ulb_x_w = sam_output_ulb_x_w['low_res_logits']
                    sam_prob_ulb_x_w = torch.softmax(sam_logits_ulb_x_w, dim=1)
                    sam_prob, sam_pseudo_label = torch.max(sam_prob_ulb_x_w, dim=1)
                    unet_size_sam_prob_ulb_x_w = F.interpolate(sam_prob_ulb_x_w, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                    unet_logits_ulb_x_w = ema_unet_model(ulb_unet_size_x_w)
                    unet_prob_ulb_x_w = torch.softmax(unet_logits_ulb_x_w, dim=1)
                    unet_prob, unet_pseudo_label = torch.max(unet_prob_ulb_x_w, dim=1)
                    
                unet_stu_output_ulb_x_w = unet_model(ulb_unet_size_x_w)
                unet_stu_prob_ulb_x_w = torch.softmax(unet_stu_output_ulb_x_w, dim=1)
                unet_stu_prob, unet_stu_pseudo_label = torch.max(unet_stu_prob_ulb_x_w, dim=1)
                low_res_unet_pseudo_label = F.interpolate(unet_pseudo_label.unsqueeze(0).float(), size=(low_res, low_res), mode='nearest').long().squeeze(0)
                if args.dataset == 'fundus':
                    self_conf = dice_calcu[args.dataset](np.asarray(to_2d(unet_stu_pseudo_label).cpu()), to_2d(unet_pseudo_label).cpu(), ret_arr=True)
                    mutual_conf = dice_calcu[args.dataset](np.asarray(to_2d(low_res_unet_pseudo_label).cpu()), to_2d(sam_pseudo_label).cpu(), ret_arr=True)
                else:
                    self_conf = dice_calcu[args.dataset](np.asarray(unet_stu_pseudo_label.clone().cpu()), unet_pseudo_label.clone().cpu(), ret_arr=True)
                    mutual_conf = dice_calcu[args.dataset](np.asarray(low_res_unet_pseudo_label.clone().cpu()), sam_pseudo_label.clone().cpu(), ret_arr=True)
                self_conf, mutual_conf = np.mean(self_conf, axis=0), np.mean(mutual_conf, axis=0)
                
                self_conf_sta.update(np.mean(self_conf))
                mutual_conf_sta.update(np.mean(mutual_conf))
                ratio_sta.update(np.mean(self_conf * mutual_conf))
                
                ratio =  torch.tensor(self_conf * mutual_conf).view(len(ulb_x_w),1,1,1).cuda()
                unet_size_prob_ulb_x_w = (1-ratio)*unet_size_sam_prob_ulb_x_w + ratio*unet_prob_ulb_x_w
                unet_size_prob, unet_size_pseudo_label = torch.max(unet_size_prob_ulb_x_w, dim=1)
                unet_size_mask = (unet_size_prob > threshold).unsqueeze(1).float()
                low_res_prob_ulb_x_w = F.interpolate(unet_size_prob_ulb_x_w, size=(low_res, low_res), mode='bilinear', align_corners=False)
                low_res_prob, low_res_pseudo_label = torch.max(low_res_prob_ulb_x_w, dim=1)
                low_res_mask = (low_res_prob > threshold).unsqueeze(1).float()
                
                unet_size_label_box = torch.stack([obtain_cutmix_box(img_size=patch_size, p=1.0) for i in range(len(ulb_x_s))], dim=0)
                unet_size_img_box = unet_size_label_box.unsqueeze(1)
                img_box = F.interpolate(unet_size_img_box, size=(args.img_size, args.img_size), mode='nearest')
                low_res_img_box = F.interpolate(unet_size_img_box, size=(low_res, low_res), mode='nearest')
                low_res_label_box = low_res_img_box.squeeze(1)
                ulb_unet_size_x_s_ul = ulb_unet_size_x_s * (1-unet_size_img_box) + lb_unet_size_x_w * unet_size_img_box
                ulb_x_s_ul = ulb_x_s * (1-img_box) + lb_x_w * img_box
                unet_size_mask[unet_size_img_box.expand(unet_size_mask.shape) == 1] = 1
                low_res_mask[low_res_img_box.expand(low_res_mask.shape) == 1] = 1
                low_res_pseudo_label_ul = (low_res_pseudo_label * (1-low_res_label_box) + lb_low_res_mask * low_res_label_box).long()
                unet_size_pseudo_label_ul = (unet_size_pseudo_label * (1-unet_size_label_box) + lb_unet_size_mask * unet_size_label_box).long()
                
                sam_output_lb_x_w = SAM_model(lb_x_w, multimask_output, args.img_size)
                sam_logits_lb_x_w = sam_output_lb_x_w['low_res_logits']
                sam_output_ulb_x_s_ul = SAM_model(ulb_x_s_ul, multimask_output, args.img_size)
                sam_logits_ulb_x_s_ul = sam_output_ulb_x_s_ul['low_res_logits']
                unet_logits_lb_x_w = unet_model(lb_unet_size_x_w)
                unet_logits_ulb_x_s_ul = unet_model(ulb_unet_size_x_s_ul)
                
                sam_prob_ulb_x_s_ul = torch.softmax(sam_logits_ulb_x_s_ul, dim=1)
                unet_size_sam_prob_ulb_x_s_ul = F.interpolate(sam_prob_ulb_x_s_ul, size=(patch_size, patch_size), mode='nearest')
                unet_prob_ulb_x_s_ul = torch.softmax(unet_logits_ulb_x_s_ul, dim=1)
                _, sam_PL_stu = torch.max(unet_size_sam_prob_ulb_x_s_ul, dim=1)
                _, unet_PL_stu = torch.max(unet_prob_ulb_x_s_ul, dim=1)
                cons_mask = (sam_PL_stu == unet_PL_stu).unsqueeze(1).float()
                low_res_cons_mask = F.interpolate(cons_mask, size=(low_res, low_res), mode='nearest')
                discons_mask = 1-cons_mask
                low_res_discons_mask = 1-low_res_cons_mask
                epsilon = 1e-10
                sam_prob_ulb_x_s_ul = torch.clamp(sam_prob_ulb_x_s_ul, epsilon, 1)
                unet_prob_ulb_x_s_ul = torch.clamp(unet_prob_ulb_x_s_ul, epsilon, 1)
                cons_loss = (-(sam_prob_ulb_x_s_ul*torch.log(sam_prob_ulb_x_s_ul)*low_res_cons_mask).mean()-(unet_prob_ulb_x_s_ul*torch.log(unet_prob_ulb_x_s_ul)*cons_mask).mean())/2
                discons_loss = ((unet_size_sam_prob_ulb_x_s_ul-unet_prob_ulb_x_s_ul)**2*discons_mask).mean()

                if args.dataset == 'fundus':
                    sam_pseudo_label_2layer = to_2d(sam_pseudo_label)
                    unet_pseudo_label_2layer = to_2d(unet_pseudo_label)
                    pseudo_label_2layer = to_2d(unet_size_pseudo_label)
                    ulb_low_res_mask_2layer = to_2d(ulb_low_res_mask)
                    ulb_unet_size_mask_2layer = to_2d(ulb_unet_size_mask)
                    sam_ulb_dice = dice_calcu[args.dataset](np.asarray(sam_pseudo_label_2layer.cpu()), ulb_low_res_mask_2layer.cpu())
                    unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label_2layer.cpu()), ulb_unet_size_mask_2layer.cpu())
                    ulb_dice = dice_calcu[args.dataset](np.asarray(pseudo_label_2layer.cpu()), ulb_unet_size_mask_2layer.cpu())
                else:
                    sam_ulb_dice = dice_calcu[args.dataset](np.asarray(sam_pseudo_label.cpu()), ulb_low_res_mask.cpu())
                    unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label.cpu()), ulb_unet_size_mask.cpu())
                    ulb_dice = dice_calcu[args.dataset](np.asarray(unet_size_pseudo_label.cpu()), ulb_unet_size_mask.cpu())
                for n, p in enumerate(part):
                    unet_ulb_dice_sta[n].update(unet_ulb_dice[n])
                    SAM_ulb_dice_sta[n].update(sam_ulb_dice[n])
                    ensemble_ulb_dice_sta[n].update(ulb_dice[n])

                sam_sup_loss = ce_loss(sam_logits_lb_x_w, lb_low_res_mask).mean() + \
                            dice_loss(sam_logits_lb_x_w, lb_low_res_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
                unet_sup_loss = ce_loss(unet_logits_lb_x_w, lb_unet_size_mask).mean() + \
                            dice_loss(unet_logits_lb_x_w, lb_unet_size_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                consistency_weight = get_current_consistency_weight(
                    iter_num // (args.max_iterations/args.consistency_rampup))

                sam_unsup_loss = (ce_loss(sam_logits_ulb_x_s_ul, low_res_pseudo_label_ul) * low_res_mask.squeeze(1)).mean() + \
                                dice_loss(sam_logits_ulb_x_s_ul, low_res_pseudo_label_ul.unsqueeze(1), mask=low_res_mask, softmax=softmax, sigmoid=sigmoid, multi=multi)
                unet_unsup_loss = (ce_loss(unet_logits_ulb_x_s_ul, unet_size_pseudo_label_ul) * unet_size_mask.squeeze(1)).mean() + \
                                dice_loss(unet_logits_ulb_x_s_ul, unet_size_pseudo_label_ul.unsqueeze(1), mask=unet_size_mask, softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                loss = sam_sup_loss + unet_sup_loss + consistency_weight * (sam_unsup_loss + unet_unsup_loss+cons_loss+discons_loss)

            sam_optimizer.zero_grad()
            unet_optimizer.zero_grad()

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(sam_optimizer)
                scaler.step(unet_optimizer)
                scaler.update()
            else:
                loss.backward()
                sam_optimizer.step()
                unet_optimizer.step()

            update_SAM_ema_variables(SAM_model, ema_SAM_model, args.ema_decay, iter_num)
            update_Unet_ema_variables(unet_model, ema_unet_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            for n, p in enumerate(part):
                text = 'train/unet_ulb_{}_dice'.format(p)
                writer.add_scalar(text, unet_ulb_dice[n], iter_num)
            for n, p in enumerate(part):
                text = 'train/sam_ulb_{}_dice'.format(p)
                writer.add_scalar(text, sam_ulb_dice[n], iter_num)
            for n, p in enumerate(part):
                text = 'train/ensemble_ulb_{}_dice'.format(p)
                writer.add_scalar(text, ulb_dice[n], iter_num)
            writer.add_scalar('train/self_conf', np.mean(self_conf), iter_num)
            writer.add_scalar('train/mutual_conf', np.mean(mutual_conf), iter_num)
            writer.add_scalar('train/ratio', np.mean(self_conf*mutual_conf), iter_num)
            writer.add_scalar('train/mask', unet_size_mask.mean(), iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/sam_sup_loss', sam_sup_loss.item(), iter_num)
            writer.add_scalar('train/sam_unsup_loss', sam_unsup_loss.item(), iter_num)
            writer.add_scalar('train/unet_sup_loss', unet_sup_loss.item(), iter_num)
            writer.add_scalar('train/unet_unsup_loss', unet_unsup_loss.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            if p_bar is not None:
                p_bar.update()

            if args.dataset == 'fundus':
                p_bar.set_description('iter %d:L:%.4f,sSL:%.4f,sUL:%.4f,uSL:%.4f,uUL:%.4f,%.4f,%.4f,cons:%.4f,mask:%.4f,sd:%.4f,%.4f,ud:%.4f,%.4f,d:%.4f,%.4f,s_m_r:%.4f,%.4f,%.4f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), cons_loss.item(), discons_loss.item(), consistency_weight, 
                                           unet_size_mask.mean(), sam_ulb_dice[0], sam_ulb_dice[1], unet_ulb_dice[0], unet_ulb_dice[1], ulb_dice[0], ulb_dice[1], 
                                           np.mean(self_conf), np.mean(mutual_conf), np.mean(self_conf*mutual_conf)))
            elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                p_bar.set_description('iter %d: L:%.4f, sSL: %.4f, sUL: %.4f, uSL: %.4f, uUL: %.4f,%.4f,%.4f, cons: %.4f, mask: %.4f, sd: %.4f, ud: %.4f, d: %.4f,s_m_r:%.4f,%.4f,%.4f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), cons_loss.item(), discons_loss.item(), consistency_weight, 
                                           unet_size_mask.mean(), sam_ulb_dice[0], unet_ulb_dice[0], ulb_dice[0], 
                                           np.mean(self_conf), np.mean(mutual_conf), np.mean(self_conf*mutual_conf)))
            elif args.dataset == 'MNMS':
                p_bar.set_description('iter %d:L:%.4f,sSL:%.4f,sUL:%.4f,uSL:%.4f,uUL:%.4f,%.4f,%.4f,cons:%.4f,mask:%.4f,sd:%.4f,%.4f,%.4f,ud:%.4f,%.4f,%.4f,d:%.4f,%.4f,%.4f,s_m_r:%.4f,%.4f,%.4f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), cons_loss.item(), discons_loss.item(), consistency_weight, 
                                           unet_size_mask.mean(), sam_ulb_dice[0], sam_ulb_dice[1], sam_ulb_dice[2], unet_ulb_dice[0], unet_ulb_dice[1], unet_ulb_dice[2], ulb_dice[0], ulb_dice[1], ulb_dice[2],
                                           np.mean(self_conf), np.mean(mutual_conf), np.mean(self_conf*mutual_conf)))
            if iter_num % args.num_eval_iter == 0:
                if args.dataset == 'fundus':
                    logging.info('iteration %d : loss : %f, sam_sup_loss : %f, sam_unsup_loss : %f, unet_sup_loss : %f, unet_unsup_loss : %f, cons_w : %f, mask_ratio : %f, sd:%.6f,%.6f,ud:%.6f,%.6f,d:%.6f,%.6f,s_m_r:%.6f,%.6f,%.6f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), consistency_weight, 
                                        unet_size_mask.mean(), SAM_ulb_dice_sta[0].avg, SAM_ulb_dice_sta[1].avg, unet_ulb_dice_sta[0].avg, unet_ulb_dice_sta[1].avg, ensemble_ulb_dice_sta[0].avg, ensemble_ulb_dice_sta[1].avg, self_conf_sta.avg, mutual_conf_sta.avg, ratio_sta.avg))
                elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                    logging.info('iteration %d : loss : %f, sam_sup_loss : %f, sam_unsup_loss : %f, unet_sup_loss : %f, unet_unsup_loss : %f, cons_w : %f, mask_ratio : %f, sd:%.6f,ud:%.6f,d:%.6f,s_m_r:%.6f,%.6f,%.6f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), consistency_weight, 
                                        unet_size_mask.mean(), SAM_ulb_dice_sta[0].avg, unet_ulb_dice_sta[0].avg, ensemble_ulb_dice_sta[0].avg, self_conf_sta.avg, mutual_conf_sta.avg, ratio_sta.avg))
                elif args.dataset == 'MNMS':
                    logging.info('iteration %d : loss : %f, sam_sup_loss : %f, sam_unsup_loss : %f, unet_sup_loss : %f, unet_unsup_loss : %f, cons_w : %f, mask_ratio : %f, sd:%.6f,%.6f,%.6f,ud:%.6f,%.6f,%.6f,d:%.6f,%.6f,%.6f,s_m_r:%.6f,%.6f,%.6f' 
                                        % (iter_num, loss.item(), sam_sup_loss.item(), sam_unsup_loss.item(), unet_sup_loss.item(), unet_unsup_loss.item(), consistency_weight, 
                                        unet_size_mask.mean(), SAM_ulb_dice_sta[0].avg, SAM_ulb_dice_sta[1].avg, SAM_ulb_dice_sta[2].avg, unet_ulb_dice_sta[0].avg, unet_ulb_dice_sta[1].avg, unet_ulb_dice_sta[2].avg, ensemble_ulb_dice_sta[0].avg, ensemble_ulb_dice_sta[1].avg, ensemble_ulb_dice_sta[2].avg, self_conf_sta.avg, mutual_conf_sta.avg, ratio_sta.avg))
                text = ''
                for n, p in enumerate(part):
                    text += 'sam_ulb_%s_dice:%f' % (p, SAM_ulb_dice_sta[n].avg)
                    text += ', '
                for n, p in enumerate(part):
                    text += 'unet_ulb_%s_dice:%f' % (p, unet_ulb_dice_sta[n].avg)
                    text += ', '
                for n, p in enumerate(part):
                    text += 'ulb_%s_dice:%f' % (p, ensemble_ulb_dice_sta[n].avg)
                    if n != n_part-1:
                        text += ', '
                logging.info(text)

        if p_bar is not None:
            p_bar.close()


        logging.info('test unet model')
        text = ''
        val_dice = test(args, unet_model, test_dataloader, epoch_num+1, writer, model_name='unet')
        for n, p in enumerate(part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]
                best_dice_iter[n] = iter_num
            text += 'val_%s_best_dice: %f at %d iter' % (p, best_dice[n], best_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best_avg_dice:
            best_avg_dice = sum(val_dice) / len(val_dice)
            best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                dice_of_best_avg[n] = val_dice[n]
            save_text = "unet_avg_dice_best_model.pth"
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg unet model to {}'.format(save_best))
            if args.save_model:
                torch.save(unet_model.state_dict(), save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (best_avg_dice, best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best_avg[n])
        logging.info(text)
        logging.info('test sam model')
        stu_val_dice = test(args, SAM_model, test_dataloader, epoch_num+1, writer, model_name='SAM')
        text = ''
        for n, p in enumerate(part):
            if stu_val_dice[n] > stu_best_dice[n]:
                stu_best_dice[n] = stu_val_dice[n]
                stu_best_dice_iter[n] = iter_num
            text += 'stu_val_%s_best_dice: %f at %d iter' % (p, stu_best_dice[n], stu_best_dice_iter[n])
            text += ', '
        if sum(stu_val_dice) / len(stu_val_dice) > stu_best_avg_dice:
            stu_best_avg_dice = sum(stu_val_dice) / len(stu_val_dice)
            stu_best_avg_dice_iter = iter_num
            for n, p in enumerate(part):
                stu_dice_of_best_avg[n] = stu_val_dice[n]
            save_text = "SAM_avg_dice_best_model.pth"
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg SAM model to {}'.format(save_best))
            if args.save_model:
                try:
                    SAM_model.save_lora_parameters(save_best)
                except:
                    SAM_model.module.save_lora_parameters(save_best)
        text += 'val_best_avg_dice: %f at %d iter' % (stu_best_avg_dice, stu_best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, stu_dice_of_best_avg[n])
        logging.info(text)
    writer.close()


if __name__ == "__main__":
    if len(args.save_name) == 0:
        args.save_name = f'fixmatch_{args.model}{args.img_size}_CP_lb{args.lb_num}_dm{args.lb_domain}'
    snapshot_path = "../model/" + args.dataset + f"/{sys.argv[0].split('.')[0]}/" + args.save_name + "/"
    
    if args.dataset == 'fundus':
        train_data_path='../../data/Fundus'
        part = ['cup', 'disc']
        dataset = FundusSegmentation
        num_channels = 3
        patch_size = 256
        num_classes = 2
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        args.max_iterations = 30000
        if args.domain_num >=4:
            args.domain_num = 4
    elif args.dataset == 'prostate':
        train_data_path="../../data/ProstateSlice"
        num_channels = 1
        patch_size = 384
        num_classes = 1
        part = ['base'] 
        dataset = ProstateSegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 255
        args.max_iterations = 60000
        if args.domain_num >= 6:
            args.domain_num = 6
    elif args.dataset == 'MNMS':
        train_data_path="../../data/mnms"
        num_channels = 1
        patch_size = 288
        num_classes = 3
        part = ['lv', 'myo', 'rv'] 
        dataset = MNMSSegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 0
        args.max_iterations = 60000
        if args.domain_num >= 4:
            args.domain_num = 4
    elif args.dataset == 'BUSI':
        train_data_path="../../data/Dataset_BUSI_with_GT"
        num_channels = 1
        patch_size = 256
        num_classes = 1
        part = ['base'] 
        dataset = BUSISegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 0
        args.max_iterations = 30000
        if args.domain_num >= 2:
            args.domain_num = 2
    
    if args.lb_num < 8:
        args.label_bs = 2
        args.unlabel_bs = 2
    else:
        args.label_bs = 4
        args.unlabel_bs = 4

    if num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
    n_part = len(part)
    dice_calcu = {'fundus':metrics.dice_coeff_2label, 'prostate':metrics.dice_coeff, 'MNMS':metrics.dice_coeff_3label, 'BUSI':metrics.dice_coeff}

    ckpt = {'SAM':'../../checkpoints/sam_vit_b_01ec64.pth', 'MedSAM':'../../checkpoints/medsam_vit_b.pth'}
    args.ckpt = ckpt[args.model]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copy('./{}'.format(sys.argv[0]), snapshot_path + '/{}'.format(sys.argv[0]))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    train(args, snapshot_path)
