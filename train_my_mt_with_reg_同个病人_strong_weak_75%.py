"""
    #75%标注比例下
    CPAC-Net完整体
"""
import argparse
import logging
import os
import random
import shutil
import sys
import time
from unet_model import *
from mt_Dataset import *
from predict import predict
from calzhibiao import cal_zhibiao,cal_zhibiao_hys_dantongdao,cal_zhibiao_hys_dantongdao_teshubiaozhunhua_RCPS,cal_zhibiao_hys_dantongdao_jiandanguiyihua_RCPS
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import shutil
import losses,  ramps
from mt_Dataset import (hysDataSets, TwoStreamBatchSampler,hysDataSetsdantongdao)
import losses, metrics, ramps
import cv2
# from FCN_resnet import *
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
import itertools
import glob
from augmentation_2d import SpatialTransform_2d_random,get_augment_list
# torch.autograd.set_detect_anomaly(True)
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.weight.data.fill_(1)
            m.weight.data.fill_(1)
            m.weight.data.fill_(1)
            m.weight.data.fill_(1)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

class ConfusionMatrix_new(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = np.zeros((n, n), dtype=np.int64)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k] + b[k]
            self.mat += np.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        h = self.mat
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = np.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = np.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not np.distributed.is_available():
            return
        if not np.distributed.is_initialized():
            return
        np.distributed.barrier()
        np.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.10f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.10f}').format(
            acc_global.item() * 100,
            ['{:.10f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.10f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100)

    def iou(self):
        iu = self.compute()[2]
        return iu.tolist()[1]
    def acc_global(self):
        acc_global = self.compute()[0]
        return acc_global


def get_current_consistency_weight(epoch,consistency=0.1,consistency_rampup=200):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

transform = transforms.Compose([
    transforms.ToTensor()
    # ,transforms.Resize((360,740))
])

def binary_to_onehot(labels, num_classes):
    """
    将二值标签转换为 one-hot 编码
    :param labels: 二值标签，形状为 (b, h, w)
    :param num_classes: 类别数量
    :return: one-hot 编码后的标签，形状为 (b, num_classes, h, w)
    """
    b, h, w = labels.shape
    # 将标签展平
    flat_labels = labels.view(b, -1)
    flat_labels = flat_labels.long()
    # 转换为 one-hot 编码
    onehot_flat = F.one_hot(flat_labels, num_classes=num_classes)
    # 调整形状回 (b, num_classes, h, w)
    onehot = onehot_flat.permute(0, 2, 1).reshape(b, num_classes, h, w)
    return onehot


def train(snapshot_path,zhibiao_dir,fold_num,data_file,epo_num=500,baifenbi=1,confidence_threshold=0.95):#bili用指百分之多少标注数据

    label_do_weak_augmentation = True
    unlabel_do_weak_augmentation= True
    save_weak_aug_img =False
    max_strong_augmentation_num = 2
    weak_augmentation = SpatialTransform_2d_random(do_rotation=True,
                                            # 旋转角度范围  -np.pi / 9 * (180 / np.pi) =-20
                                            # angle=(-np.pi / 9, np.pi / 9),
                                            angle=(0, np.pi),
                                            # 是否进行尺度变换
                                            do_scale=True,
                                            # 尺度变换，水平，垂直尺度因子
                                            scale=(0.80, 1.20),
                                            do_flip=True,
                                            random_percent=0.5
                                            )
    

    
    confidence_threshold = confidence_threshold
    fold_num =fold_num
    num_classes = 2
    batch_size = 1
    epo_num = epo_num
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    model_path=os.path.join(snapshot_path,data_file_name)
    if not os.path.exists(model_path):
        print("%s没有，正在创建" % model_path)
        os.makedirs(model_path)
    def create_model(n_channels, n_classes, bilinear, ema=False,aux=False):
        # model = UNet_seg_2d_RCPS_base(n_channels, n_classes)
        # 输出包含编码器第一层的特征，用于后续计算两张图像的余弦相似度
        model = UNet_seg_2d_RCPS_with_confidence(n_channels, n_classes)
        
        if ema:
            for param in model.parameters():
                param.detach_()

        return model
    logging.info(data_file_name)
    device = torch.device('cuda:1')
    model1 = create_model(1, 2, bilinear=True, ema=False,aux=False).to(device)
    # model1 = kaiming_normal_init_weight(model1)
    model2 = create_model(1, 2, bilinear=True, ema=True,aux=False).to(device)
    # model2 = kaiming_normal_init_weight(model2)
    # 配准模型
    reg_model ='CPAC_Net/Reg_Model/20240929_5folds_reg_se_2pamV1_training_info_k_0_channels_1_classes_2_epoches_200_iters_200_batch_size_1_is_aug_True_L_sim_1ssim_1ncc_1loss_sec'
    pth_files = glob.glob(os.path.join(reg_model,str(fold_num), '*.pth')) 
    
    pretrainRegmodel=pth_files[0]
    print(pretrainRegmodel)
    Reger = UNet_reg_with_se_2pam(n_channels=1).to(device)

    if(pretrainRegmodel!=''): #如果已经有预训练的配准模型，则直接加载不要进行训练了
        Reger.load_state_dict(torch.load(pretrainRegmodel))
    
 
# 
    db_train= hysDataSets_dantongdao_jiandanguiyihua_50_75percent(data_file=data_file, img_path="/mnt/hys/Datasets/重复测量（hys整理版）/img_crop_npy", zhibiao_dir=zhibiao_dir,
                            split="train",baifenbi=baifenbi,with_affine_field=True)
    db_test = hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path="/mnt/hys/Datasets/重复测量（hys整理版）/img_crop_npy", zhibiao_dir=zhibiao_dir,
                          split="test")
    db_val = hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path="/mnt/hys/Datasets/重复测量（hys整理版）/img_crop_npy", zhibiao_dir=zhibiao_dir,
                          split="val")

    total_slices = len(db_train)
    labeled_slice = total_slices//2  # 通过index 划分label&unlabel
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,worker_init_fn=3407)#pin_memory 是加快速度 内存大可以设置为True

    model1.train()
    model2.train()
    Reger.eval()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)


    optimizer1= torch.optim.AdamW(model1.parameters(), lr=0.001,weight_decay=1e-6)
    # 设置余弦退火的学习率调度器
    T_max = epo_num  # 一个周期的最大步数
    # scheduler1= CosineAnnealingLR(optimizer1, T_max=T_max, eta_min=0.00001) #训练到T_max时的学习率是eta_min
    scheduler1= CosineAnnealingLR(optimizer1, T_max=T_max, eta_min=0.0001) #训练到T_max时的学习率是eta_min
    # optimizer2 = optim.SGD(model2.parameters(), lr=0.01,
    #                        momentum=0.9)
    max_epoch = epo_num
    ce_loss = CrossEntropyLoss().to(device)
    dice_loss = losses.DiceLoss(num_classes).to(device)
    writer = SummaryWriter(snapshot_path + '/log/'+str(fold_num))
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_performance1 = 0.0
    best_performance2 = 0.0
    scaler = GradScaler()
    iterator = tqdm(range(max_epoch), ncols=70)

    num_thre=baifenbi*epo_num
    print("num_thred:",num_thre)
    
    loss_pse_label_correction_dist=[]
    loss_consistence_dist=[]
    supervised_loss_dist=[]
    loss_dist=[]
    
    # 训练开始时间
    total_train_start_time = time.time()
    # 初始化峰值显存记录
    peak_memory = 0
    # 初始化最佳性能时的总训练耗时
    best_performance_total_time = 0.0
    # 初始化最佳性能时的epoch
    best_performance_epoch = 0

    for epoch_num in iterator:
        # Epoch开始时间
        epoch_start_time = time.time()
        # 重置当前epoch的显存记录
        torch.cuda.reset_peak_memory_stats()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            
            label_image, label,label_name = sampled_batch['image'], sampled_batch['label'],sampled_batch['labelimg_name']
            label_image, label= label_image.to(device), label.to(device)
            unlabel_name=sampled_batch['unlabelname']
            unlabel_image=sampled_batch['unlabel_image'].to(device)
            # 有标签往无标签仿射的图像
            affine_img, affine_label,affine_label_onehot = sampled_batch['l_u_img'].to(device), sampled_batch['l_u_seg'].to(device),sampled_batch['l_u_seg_onehot'].to(device)
            # 有标签往无标签的变形图像以及标签
            with torch.no_grad():
                w_img_l_u,_, w_seg_l_u_onehot, _,_ = Reger(affine_img, unlabel_image, affine_label_onehot,None)
            
            # 弱数据增强
            batch = label_image.shape[0]
            # 无标签数据和变形图像数据 流进行一样的弱增强
            if(unlabel_do_weak_augmentation):
                with torch.no_grad():
                    # 增强前的图像深拷贝
                    unlabel_image_copy = unlabel_image.detach().cpu().numpy().copy()[0,0]
                    w_img_l_u_copy = w_img_l_u.detach().cpu().numpy().copy()[0,0]
                    w_seg_l_u_label_copy = w_seg_l_u_onehot.detach().cpu().numpy().copy()[0,1]
                    for b in range(0,batch):
                        code_spa = weak_augmentation.rand_coords(unlabel_image.shape[2:]).to(device)
                        # 图像进行增强
                        unlabel_image[b:b+1]  = weak_augmentation.augment_spatial(unlabel_image[b:b+1], code_spa)
                        w_img_l_u[b:b+1]  = weak_augmentation.augment_spatial(w_img_l_u[b:b+1], code_spa)

                        w_seg_l_u_onehot[b:b+1]  = weak_augmentation.augment_spatial(w_seg_l_u_onehot[b:b+1], code_spa,mode='nearest')
                        w_seg_l_u_onehot[b:b+1,0]=1-w_seg_l_u_onehot[b:b+1,1]
                    # 增强后的图像深拷贝
                    unlabel_image_copy2 = unlabel_image.detach().cpu().numpy().copy()[0,0]
                    w_img_l_u_copy2 = w_img_l_u.detach().cpu().numpy().copy()[0,0]
                    w_seg_l_u_label_copy2 = w_seg_l_u_onehot.detach().cpu().numpy().copy()[0,1]
                    
                
                
            # 标签数据流进行弱增强
            if(label_do_weak_augmentation):
                with torch.no_grad():
                    label_copy = label.detach().cpu().numpy().copy()[0]
                    # 转换成onehot编码便于增强
                    label = binary_to_onehot(label, num_classes).float()
                    label_image_copy = label_image.detach().cpu().numpy().copy()[0,0]
                    for b in range(0,batch):
                        code_spa = weak_augmentation.rand_coords(label_image.shape[2:]).to(device)
                        # 图像进行增强
                        label_image[b:b+1] = weak_augmentation.augment_spatial(label_image[b:b+1] , code_spa)
                        label[b:b+1]  = weak_augmentation.augment_spatial(label[b:b+1], code_spa,mode='nearest')
                        label[b:b+1,0]=1-label[b:b+1,1]
                        
                    # 变成二值标签
                    label = label[:,1]
                    
                    # 增强后的图像深拷贝
                    label_copy2 = label.detach().cpu().numpy().copy()[0]
                    label_image_copy2 = label_image.detach().cpu().numpy().copy()[0,0]
            if(iter_num%len(trainloader)==0 and epoch_num%100 ==0):
                if(save_weak_aug_img):
                    plt.rcParams['font.size'] = 30
                    fig, axes =plt.subplots(nrows=2,ncols=5,figsize=(50, 40))
                    axes[0,0].imshow(((unlabel_image_copy)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,0].set_title("unlabelimg_bf_Aug")
                    axes[1,0].imshow(((unlabel_image_copy2)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,0].set_title("unlabelimg_af_Aug")
                    
                    axes[0,1].imshow(((w_img_l_u_copy)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,1].set_title("warp_bf_Aug")
                    # Ai的腺体标签
                    axes[1,1].imshow(((w_img_l_u_copy2)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,1].set_title("warp_af_Aug")
                            
                            
                    axes[0,2].imshow(((w_seg_l_u_label_copy)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,2].set_title("warp_seg_bf")
                            
                    axes[1,2].imshow(((w_seg_l_u_label_copy2)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,2].set_title("warp_seg_af")
                            
                            
                    axes[0,3].imshow((label_image_copy*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,3].set_title("label_image_bf")
            
                    axes[1,3].imshow(((label_image_copy2)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,3].set_title("label_image_af")

                    axes[0,4].imshow(((label_copy)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,4].set_title("label_bf")

                    axes[1,4].imshow(((label_copy2)*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,4].set_title("label_af")

                            # 调整子图之间的间距
                    plt.tight_layout()
                    # cv2.imwrite('{}.jpg'.format(unlabel_name[0]), np.uint8(unlabel_image_copy2*255))
                    save_generate_image_file = os.path.join(snapshot_path+'/aug_compair/'+str(fold_num))
                    if not os.path.exists(save_generate_image_file):
                        os.makedirs(save_generate_image_file)
                    generate_image_name = "epoch{0}_iter{1}.jpg".format(epoch_num,iter_num)
                    plt.savefig(os.path.join(save_generate_image_file,generate_image_name))
                    plt.close()
                

                
                # save_jiegoutu_image_file = os.path.join(snapshot_path+'/jiegoutu/'+str(fold_num))
                # if not os.path.exists(save_jiegoutu_image_file):
                #     os.makedirs(save_jiegoutu_image_file)
                # generate_unlabel_img_bf_aug_name = "epoch{0}_iter{1}_unlabel_img_bf_aug.jpg".format(epoch_num,iter_num)
                # generate_unlabel_img_af_weak_aug_name = "epoch{0}_iter{1}_unlabel_img_af_weak_aug.jpg".format(epoch_num,iter_num)
                # generate_label_img_bf_aug_name = "epoch{0}_iter{1}_label_img_bf_aug.jpg".format(epoch_num,iter_num)
                # generate_label_img_af_aug_name = "epoch{0}_iter{1}_label_img_af_aug.jpg".format(epoch_num,iter_num)
                # generate_label_bf_aug_name = "epoch{0}_iter{1}_label_bf_aug.jpg".format(epoch_num,iter_num)
                # generate_label_af_aug_name = "epoch{0}_iter{1}_label_af_aug.jpg".format(epoch_num,iter_num)
                # generate_warp_bf_aug_name = "epoch{0}_iter{1}_warp_bf_aug.jpg".format(epoch_num,iter_num)
                # generate_warp_lab_bf_aug_name = "epoch{0}_iter{1}_warp_lab_bf_aug.jpg".format(epoch_num,iter_num)
                # generate_warp_lab_af_aug_name = "epoch{0}_iter{1}_warp_lab_af_aug.jpg".format(epoch_num,iter_num)
                # generate_warp_af_aug_name = "epoch{0}_iter{1}_warp_af_aug.jpg".format(epoch_num,iter_num)
                
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_unlabel_img_bf_aug_name),((unlabel_image_copy)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_unlabel_img_af_weak_aug_name),((unlabel_image_copy2)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_label_img_bf_aug_name),((label_image_copy)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_label_img_af_aug_name),((label_image_copy2)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_label_bf_aug_name),((label_copy)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_warp_bf_aug_name),((w_img_l_u_copy)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_warp_lab_bf_aug_name),((w_seg_l_u_label_copy)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_warp_lab_af_aug_name),((w_seg_l_u_label_copy2)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_label_af_aug_name),((label_copy2)*255).astype(np.uint8))
                # cv2.imwrite(os.path.join(save_jiegoutu_image_file,generate_warp_af_aug_name),((w_img_l_u_copy2)*255).astype(np.uint8))
                

            
            
            if(max_strong_augmentation_num>0):
                # 注意强增强和弱增强数据的梯度并不共享
                strong_unlabel_image = unlabel_image.detach().clone()
                with torch.no_grad():
                    for b in range(0,batch):
                        strong_augmentation_num =random.randint(0,max_strong_augmentation_num+1)
                        augment_list = get_augment_list()
                        #
                        # (img_aug_identity, None),# 原始图像
                        # (img_aug_autocontrast, None),# 随机对比度调整
                        # (img_aug_equalize, None), # 直方图均衡化
                        # (img_aug_blur, [0.1, 2.0]), #高斯模糊
                        # (img_aug_contrast, [0.05, 0.95]),# 调整图像对比度
                        # (img_aug_brightness, [0.05, 0.95]),# 调整亮度
                        # (img_aug_color, [0.05, 0.95]),# 调整图像的色彩饱和度
                        # (img_aug_sharpness, [0.05, 0.95]),# 调整图像的锐度
                        # (img_aug_posterize, [4, 8]),#对图像进行色调分离操作。
                        # (img_aug_solarize, [1, 256]),#对图像进行曝光反转（Solarize）操作。
                        # (img_aug_hue, [0, 0.5])# 调整图像的色调

                        ops = random.choices(augment_list, k=strong_augmentation_num)
                        # ops = random.choices(augment_list, k=1)
                        
                        
                        
                        for op, scales in ops:
                            # print("="*20, str(op))
                            # t=strong_unlabel_image[b]
                            strong_unlabel_image[b] = op(strong_unlabel_image[b], scales)
                        #不保存图片，
                        if(False and iter_num%len(trainloader)==0 and epoch_num%10 ==0):
                            
                            plt.rcParams['font.size'] = 30
                            fig, axes =plt.subplots(nrows=2,ncols=1,figsize=(50, 40))
                            axes[0].imshow(((unlabel_image.detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                            axes[0].set_title("unlabelimg_bf_Aug")
                            axes[1].imshow(((strong_unlabel_image.detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                            axes[1].set_title("unlabelimg_af_Aug")
                            # 调整子图之间的间距
                            plt.tight_layout()

                            # cv2.imwrite(os.path.join(save_jiegoutu_image_file,"epoch{0}_iter{1}_unlabel_img_af_weak_strong_aug.jpg".format(epoch_num,iter_num)),((strong_unlabel_image.detach().cpu().numpy()[0,0])*255).astype(np.uint8))

                            # cv2.imwrite('{}.jpg'.format(unlabel_name[0]), np.uint8(unlabel_image_copy2*255))
                            # save_generate_image_file = os.path.join(snapshot_path+"/"+str(op)+"/"+str(fold_num))
                            save_generate_image_file = os.path.join(snapshot_path+"/strong_weak_augmentation/"+str(fold_num))
                            if not os.path.exists(save_generate_image_file):
                                os.makedirs(save_generate_image_file)
                            generate_image_name = "epoch{0}_iter{1}.jpg".format(epoch_num,iter_num)
                            plt.savefig(os.path.join(save_generate_image_file,generate_image_name))
                            plt.close()
                    
                
            with autocast():
                # 有标签 弱增强
                label_outputs = model1(label_image)['output'] #未经过归一化处理的原始输出，可以是任意实数
                label_outputs_soft = torch.softmax(label_outputs, dim=1) #概率分布（0-1）
                # 无标签 弱增强
                unlabel_outputs1 = model1(unlabel_image)['output'] #未经过归一化处理的原始输出，可以是任意实数
                unlabel_outputs_soft1 = torch.softmax(unlabel_outputs1, dim=1)
                
                #编写图像以及编写标签也送入训练。 弱增强的变形标签数据
                w_img_l_u_output = model1(w_img_l_u)['output'] #未经过归一化处理的原始输出，可以是任意实数
                w_img_l_u_output_soft = torch.softmax(w_img_l_u_output, dim=1)
                
                #强增强图像
                if(max_strong_augmentation_num>0):
                    strong_unlabel_image_outputs = model1(strong_unlabel_image)['output']
                    strong_unlabel_image_outputs_soft1 = torch.softmax(strong_unlabel_image_outputs, dim=1)

                with torch.no_grad():
                    
                    unlabel_output2=model2(unlabel_image)["output"] #1,2 350 740 #预测的概率值
                    # 获取两张图像编码的第一层特征用于计算余弦相似度。
                    unlabel_image_encode_1=model2(unlabel_image)["encode_1"]
                    wrap_l_u_encode_1=model2(w_img_l_u)["encode_1"]
                preds = F.softmax(unlabel_output2, dim=1)
                # 无标签图像预测的不确定性
                uncertainty = -1.0 * \
                    torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
                consistency_weight = 0.01+0.1*get_current_consistency_weight(epoch_num,consistency_rampup=epo_num) #get_current_consistency_weight(epoch_num)最大值为0.1
                # consistency_weight = get_current_consistency_weight(epoch_num,consistency_rampup=epo_num) #get_current_consistency_weight(epoch_num)最大值为0.1
 
                # print(outputs1.shape,outputs2.shape)
                
                # 用教师模型上第一层的编码特征计算变形图像与无标签图像的余弦相似度
                sim = F.cosine_similarity(unlabel_image_encode_1.detach(), wrap_l_u_encode_1.detach(), dim=1) #1 350 740
                sim_unsqueze = sim.unsqueeze(1)
                simility_confident_mask = (sim > confidence_threshold).unsqueeze(1) # 1*1*350*740 值0或者1
                
                simility_confident_mask_np = simility_confident_mask.detach().cpu().numpy()[0][0]  
                simility_confident_mask_np = simility_confident_mask_np.astype('uint8')  # 转换为 uint8 类型
                # 这里算的的是无标签图像加噪声和没加噪声前后的一致性
                # 在训练初期，模型对数据的理解还不够准确，不确定性较高，此时较低的阈值可以让模型更关注那些不确定性较低的样本，即模型比较有把握的样本，进行学习和优化。
                # 随着训练的进行，模型的性能逐渐提升，对数据的理解也更加深入，此时逐渐升高阈值，可以让模型开始关注那些在训练初期被认为不确定性较高的样本，进一步挖掘这些样本中的信息，提高模型的泛化能力
                #  阈值从 0.75 * np.log(2) 升高到 (0.75 + 0.25) * np.log(2)。 
                #  (0.51986025，0.693147]  np.log(2)大约为0.693147
                # threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num,
                #                             max_epoch*len(trainloader)))*np.log(2)

                # print("threshold",threshold)
               
                # print(ramps.sigmoid_rampup(iter_num,
                #                             max_epoch*len(trainloader)))
                # #(0.0047，0.693147]  np.log(2)大约为0.693147
                # threshold = (0+1*ramps.sigmoid_rampup(iter_num,
                #                             max_epoch*len(trainloader)))*np.log(2)
                
                # threshold = (0.5+0.5*ramps.sigmoid_rampup(iter_num,
                #                             max_epoch*len(trainloader)))
                                
                #
                # threshold = (0.5+0.5*ramps.sigmoid_rampup(iter_num,
                #                             max_epoch*len(trainloader)*np.log(2)))
                # # [0.5023,0.84657]
                # threshold = (0.5+0.5*ramps.sigmoid_rampup(iter_num,
                #                             max_epoch*len(trainloader))*np.log(2))
                # [0.5023,0.94657]
                # ramps.sigmoid_rampup(iter_num, max_epoch*len(trainloader))*np.log(2) =0.69314
                # threshold = (0.6+0.5*ramps.sigmoid_rampup(iter_num,
                #                             max_epoch*len(trainloader))*np.log(2))
                # [0.6,0.877]
                threshold = (0.6+0.4*ramps.sigmoid_rampup(iter_num,
                                            max_epoch*len(trainloader))*np.log(2))
                #  0.75 * np.log(2) 升高到 (0.75 + 0.25) * np.log(2)。 
                # # (0.51986025，0.693147] 
                # threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num,
                #                             max_epoch*len(trainloader)))*np.log(2)
                # print("threshold",threshold)
                # 高于阈值的“确定”掩码
                certain_mask = (uncertainty < threshold) # 1*1*350*740 值false或者true
                # 低于阈值的不确定性掩码，与置信度高的区域进行相与，得到相对可信补充。将补充的掩码叠加到certain_mask中得到生成的掩码
                uncertain_mask = (uncertainty >= threshold) # 1*1*350*740 值false或者true
                # 教师模型预测不确定的像素用配准的标签中高置信度的像素代替。得到的是需要补充的点。
                buchong_mask=torch.logical_and(simility_confident_mask, uncertain_mask)# 1*1*350*740 值false或者true

                generate_mask = (torch.logical_or(certain_mask, buchong_mask)).float()
                # 判断是否有大于 1 的值
                # has_greater_than_one = torch.any(generate_mask > 1)
                
                # print(has_greater_than_one.item())
                # has_less_than_zero = torch.any(generate_mask < 0)
                
                # print(has_less_than_zero.item())
                # 生成的标签由教师模型预测的确定标签和配准得到的可信标签组成。
                # preds 获得的是预测的概率，w_seg_l_u_onehot的值是0或者1.不能简单相加,可以用满足阈值的置信度作为该点的预测概率
                # 该值必须在0-1之间
                

                # w_seg_l_u_onehot1 = (w_seg_l_u_onehot.detach().cpu().numpy()[0,0])*255
                # w_seg_l_u_onehot1 = w_seg_l_u_onehot1.astype('uint8')  # 转换为 uint8 类型
                
                # w_seg_l_u_onehot2 = (w_seg_l_u_onehot.detach().cpu().numpy()[0,1])*255
                # w_seg_l_u_onehot2 = w_seg_l_u_onehot2.astype('uint8')  # 转换为 uint8 类型
                # cv2.imwrite('/mnt/hys/w_seg_l_u_onehot1.png', w_seg_l_u_onehot1)
                # cv2.imwrite('/mnt/hys/w_seg_l_u_onehot2.png', w_seg_l_u_onehot2)

                
                # 将另一个通道补充。使一个像素点的两通道相加为1
                result=w_seg_l_u_onehot*sim_unsqueze
                # 分离通道
                background_channel = result[:, 0:1, :, :]
                gland_channel = result[:, 1:2, :, :]

                # 计算互补值
                background_complement = 1 - gland_channel
                gland_complement = 1 - background_channel

                # 补齐通道
                background_channel = torch.where(background_channel == 0, background_complement, background_channel)
                gland_channel = torch.where(gland_channel == 0, gland_complement, gland_channel)

                # 合并通道
                final_sim_unsqueze = torch.cat([background_channel, gland_channel], dim=1)
                                
                # t =buchong_mask.float()*w_seg_l_u_onehot*sim_unsqueze
                
                t =buchong_mask.float()*final_sim_unsqueze
                
                t2=(certain_mask.float()*preds)
                
                
                generate_label = certain_mask.float()*preds+buchong_mask.float()*final_sim_unsqueze
                
                generate_label_soft = torch.softmax(generate_label,dim=1)
                generate_label_argmax = torch.argmax(generate_label_soft, dim=1)
                #强制不保存图片
                if(False and iter_num%len(trainloader)==0 and epoch_num%100 ==0):
                # if(1):
                    plt.rcParams['font.size'] = 30
                    fig, axes =plt.subplots(nrows=2,ncols=10,figsize=(50, 40))
                    axes[0,0].imshow(((unlabel_image[0][0].detach().cpu().numpy())*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,0].set_title("unlabelimg")
                    axes[1,0].imshow(((label_image[0][0].detach().cpu().numpy())*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,0].set_title("labelimg")
                    # Ai图像
                    axes[0,1].imshow(((w_img_l_u[0][0].detach().cpu().numpy())*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,1].set_title("l_u")
                    # Ai的腺体标签
                    axes[1,1].imshow(((label[0].detach().cpu().numpy())*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,1].set_title("label_gland")
                    
                    
                    axes[0,2].imshow(((w_seg_l_u_onehot[0][1].detach().cpu().numpy())*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,2].set_title("warp_gland")
                    
                    axes[1,2].imshow(((w_seg_l_u_onehot[0][0].detach().cpu().numpy())*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,2].set_title("warp_bg")
                    
                    
                    axes[0,3].imshow((((certain_mask).detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,3].set_title("certain_mask")
    
                    axes[1,3].imshow((((simility_confident_mask).detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,3].set_title("simility_confident_mask")

                    axes[0,4].imshow((((buchong_mask).detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,4].set_title("buchong_mask")
                    # data = (((generate_mask).detach().cpu().numpy()[0, 0]) * 255).astype(np.uint8)
                    # print("数据类型:", data.dtype)
                    # print("数据最小值:", data.min())
                    # print("数据最大值:", data.max())
                    axes[1,4].imshow((((generate_mask).detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,4].set_title("generate_mask")
                    # 
                    axes[0,5].imshow((((t).detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,5].set_title("buchong_bg_label")
                    

                    axes[1,5].imshow((((t).detach().cpu().numpy()[0,1])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,5].set_title("buchong_gland_label")
                    
                    axes[0,6].imshow((((torch.argmax(preds,dim=1)).detach().cpu().numpy()[0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,6].set_title("pred_seg")
                    axes[1,6].imshow((((torch.argmax(t2,dim=1)).detach().cpu().numpy()[0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,6].set_title("certain_seg")
                    
                    axes[0,7].imshow((((generate_label).detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,7].set_title("generate_bg")
                                    
                    axes[1,7].imshow((((generate_label).detach().cpu().numpy()[0,1])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,7].set_title("generate_gland")
                    axes[0,8].imshow((((generate_label_soft).detach().cpu().numpy()[0,0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,8].set_title("generate_bg_soft")
                                
                    axes[1,8].imshow((((generate_label_soft).detach().cpu().numpy()[0,1])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[1,8].set_title("generate_gland_soft")
                
                    axes[0,9].imshow((((generate_label_argmax).detach().cpu().numpy()[0])*255).astype(np.uint8),cmap="gray", vmin=0, vmax=255)
                    axes[0,9].set_title("generate_label_argmax")
                                
                    axes[1,9].axis('off') 
                    # 调整子图之间的间距
                    plt.tight_layout()
                    # plt.show()
                    # 保存图像
                    save_generate_image_file = os.path.join(snapshot_path+'/generate_image/'+str(fold_num))
                    if not os.path.exists(save_generate_image_file):
                        os.makedirs(save_generate_image_file)
                    generate_image_name = "epoch{0}_iter{1}.jpg".format(epoch_num,iter_num)
                    plt.savefig(os.path.join(save_generate_image_file,generate_image_name))
                    plt.close()
                    
                    # save_jiegoutu_image_file = os.path.join(snapshot_path+'/jiegoutu/'+str(fold_num))

                #     cv2.imwrite(os.path.join(save_jiegoutu_image_file,"epoch{0}_iter{1}_Tea_pred.jpg".format(epoch_num,iter_num)),(((torch.argmax(preds,dim=1)).detach().cpu().numpy()[0])*255).astype(np.uint8))
                #     cv2.imwrite(os.path.join(save_jiegoutu_image_file,"epoch{0}_iter{1}_pseudo_generate.jpg".format(epoch_num,iter_num)),(((generate_label_argmax).detach().cpu().numpy()[0])*255).astype(np.uint8))
                #     cv2.imwrite(os.path.join(save_jiegoutu_image_file,"epoch{0}_iter{1}_w_lab_img_Stu_pred.jpg".format(epoch_num,iter_num)),(((torch.argmax(label_outputs_soft,dim=1)).detach().cpu().numpy()[0])*255).astype(np.uint8))
                #     cv2.imwrite(os.path.join(save_jiegoutu_image_file,"epoch{0}_iter{1}_warp_img_Stu_pred.jpg".format(epoch_num,iter_num)),(((torch.argmax(w_img_l_u_output_soft,dim=1)).detach().cpu().numpy()[0])*255).astype(np.uint8))
                #     cv2.imwrite(os.path.join(save_jiegoutu_image_file,"epoch{0}_iter{1}_ws_unlab_img_Stu_pred.jpg".format(epoch_num,iter_num)),(((torch.argmax(strong_unlabel_image_outputs_soft1,dim=1)).detach().cpu().numpy()[0])*255).astype(np.uint8))
                #     cv2.imwrite(os.path.join(save_jiegoutu_image_file,"epoch{0}_iter{1}_w_unlab_img_Stu_pred.jpg".format(epoch_num,iter_num)),(((torch.argmax(unlabel_outputs_soft1,dim=1)).detach().cpu().numpy()[0])*255).astype(np.uint8))                
                # # consistency_dist = losses.softmax_mse_loss(unlabel_outputs_soft1, generate_label)
                # 定义交叉熵损失函数
                criterion = nn.CrossEntropyLoss(reduction='none').to(device)#会自动对模型的原始输出进行softmax，将其转化成概率分布在计算ce损失
                # 计算交叉熵损失，输入的是未归一化的原始输出和类别标签(不是onehot)
                pse_label_correction_dist = criterion(unlabel_outputs1,generate_label_argmax ) #
                # consistency_dist = losses.softmax_mse_loss(unlabel_outputs_soft1, generate_label_soft)

                # print(torch.sum(generate_mask==1))
                
                pse_label_correction_loss = torch.sum(
                generate_mask*pse_label_correction_dist)/(2*torch.sum(generate_mask)+1e-16)
                loss_dice = dice_loss(
                    label_outputs, label.unsqueeze(1))
                loss_ce = ce_loss(label_outputs,
                              label.long())

                # supervised_loss=loss_dice+loss_ce
                # supervised_loss=loss_dice+0.5*loss_ce
                # 变形图像送入学生模型计算有监督的损失
                w_seg_l_u_label = w_seg_l_u_onehot[:,1,:,:]
                warp_loss_dice = dice_loss(
                    w_img_l_u_output, w_seg_l_u_label.unsqueeze(1))
                warp_loss_ce = ce_loss(w_img_l_u_output,
                              w_seg_l_u_label.long())
                supervised_loss=loss_dice+0.5*loss_ce+0.5*(warp_loss_dice+0.5*warp_loss_ce)
                consistence_loss=0
                if(max_strong_augmentation_num>0):
                    # # KL 散度：是不对称的，即 KL (P||Q) ≠ KL (Q||P)，
                    # # 这意味着在使用 KL 散度时，需要明确哪个分布是参考分布，哪个是被比较的分布，其物理意义和应用场景会因参考分布的不同而有所差异。
                    kl_loss = nn.KLDivLoss(reduction='mean')
                    # # 强增强作为比较分布，弱增强的预测作为参考分布
                    consistence_loss = kl_loss(torch.log(strong_unlabel_image_outputs_soft1), unlabel_outputs_soft1)
                    # # # 弱增强作为比较分布，强增强的预测作为参考分布
                    # consistence_loss2 = kl_loss(torch.log(unlabel_outputs_soft1), strong_unlabel_image_outputs_soft1)
                    
                    # consistence_loss = (consistence_loss1+consistence_loss2)/2.0
                    
                    #MSE损失
                    # consistence_loss = losses.softmax_mse_loss(unlabel_outputs_soft1, strong_unlabel_image_outputs_soft1)
                    
                    loss_consistence_dist.append(consistence_loss.detach().cpu()) 
                     
                supervised_loss_dist.append(supervised_loss.detach().cpu())
                loss_pse_label_correction_dist.append(pse_label_correction_loss.detach().cpu())  
                
            pseudo_correction_weight = consistency_weight
            loss = supervised_loss + pseudo_correction_weight * pse_label_correction_loss+consistency_weight*consistence_loss
            # loss=loss1
            loss_dist.append(loss.detach().cpu())
        

                


            optimizer1.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer1)

            scaler.update()
            update_ema_variables(model1, model2, 0.99, iter_num)

            iter_num = iter_num + 1
            
            # 检查当前显存使用情况
            current_memory = torch.cuda.max_memory_cached() / 1024**2  # 转换为MB
            if current_memory > peak_memory:
                peak_memory = current_memory

            lr_=optimizer1.param_groups[0]['lr']
            # writer.add_scalar('info/lr', lr_, epoch_num)
            # writer.add_scalar('info/model1_total_loss', loss, epoch_num)
            # writer.add_scalar('info/supervised_loss', supervised_loss, epoch_num)
            # writer.add_scalar('info/consistence_loss',
            #                   consistence_loss, epoch_num)
            # writer.add_scalar('info/pse_label_correction_loss',
            #                   pse_label_correction_loss, epoch_num)
            # writer.add_scalar('info/consistency_weight',
            #                   consistency_weight, epoch_num)
            # writer.add_scalar('info/uncertainty',
            #                   threshold, epoch_num)
            # logging.info(
            #     'iteration %d : model1 loss : %f model2 loss : %f  pseudo_weight: %f lr:%f contour_loss1:%f contour_loss2:%f ' % (iter_num, model1_loss.item(), model2_loss.item(),consistency_weight,lr_,contour_loss1,contour_loss2))
            logging.info(
                'iteration %d : loss : %f   consistency_weight: %f lr:%f  ' % (iter_num, loss.item(), consistency_weight,lr_))
            # logging.info(
            #     'iteration %d : model1 loss : %f ' % (iter_num, loss.item()))

            if iter_num%len(trainloader)==0:
                scheduler1.step()
                # scheduler2.step()
                with torch.no_grad():
                    model1.eval()
                    model2.eval()
                    iou_val1=[]
                    iou_val2=[]
                    avg_iou_val1=0.0
                    avg_iou_val2=0.0
                    iou_test1=[]
                    iou_test2=[]
                    avg_iou_test1=0.0
                    avg_iou_test2=0.0
                    # for i_batch, sampled_batch in enumerate(valloader):
                    #     volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    #     volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
                    #     optimizer1.zero_grad()
                    #     # optimizer2.zero_grad()
                    #     output1 = model1(volume_batch)["output"]
                    #     output2 = model2(volume_batch)["output"]
                    #     output1_np = output1.cpu().detach().numpy().copy()  
                    #     output1_np = np.argmax(output1_np, axis=1)
                    #     output2_np = output2.cpu().detach().numpy().copy()  
                    #     output2_np = np.argmax(output2_np, axis=1)
                    #     label_batch_np=label_batch.cpu().detach().numpy().copy()
                    #     #------------计算val iou---------#
                    #     for j in range(output1_np.shape[0]):
                    #         matrix_val1 = ConfusionMatrix_new(2)
                    #         img1, img_label = output1_np[j], label_batch_np[j]
                    #         matrix_val1.update(img1, img_label)
                    #         iou_val1.append(matrix_val1.iou())

                    #     for j in range(output2_np.shape[0]):
                    #         matrix_val2 = ConfusionMatrix_new(2)
                    #         img2, img_label = output2_np[j], label_batch_np[j]
                    #         matrix_val2.update(img2, img_label)
                    #         iou_val2.append(matrix_val2.iou())
                    # avg_iou_val1=sum(iou_val1)/len(iou_val1)
                    # avg_iou_val2=sum(iou_val2)/len(iou_val2)
                    
                    # test
                    for i_batch, sampled_batch in enumerate(testloader):
                        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

                        # output1 = model1(volume_batch)["output"]
                        output2 = model2(volume_batch)["output"]
                        # output1_np = output1.cpu().detach().numpy().copy()  
                        # output1_np = np.argmax(output1_np, axis=1)
                        output2_np = output2.cpu().detach().numpy().copy()  
                        output2_np = np.argmax(output2_np, axis=1)
                        label_batch_np=label_batch.cpu().detach().numpy().copy()
                        #------------计算test iou---------#
                        # for j in range(output1_np.shape[0]):
                        #     matrix_test1 = ConfusionMatrix_new(2)
                        #     img1, img_label = output1_np[j], label_batch_np[j]
                        #     matrix_test1.update(img1, img_label)
                        #     iou_test1.append(matrix_test1.iou())

                        for j in range(output2_np.shape[0]):
                            matrix_test2 = ConfusionMatrix_new(2)
                            img2, img_label = output2_np[j], label_batch_np[j]
                            matrix_test2.update(img2, img_label)
                            iou_test2.append(matrix_test2.iou())
                    # avg_iou_test1=sum(iou_test1)/len(iou_test1)
                    avg_iou_test2=sum(iou_test2)/len(iou_test2)

                    # if avg_iou_val1 > best_performance1:
                    #     logging.info("model1"+str(iou_val1))
                    #     best_model1=model1
                    #     best_test_iou1=avg_iou_test1
                    #     best_performance1 = avg_iou_val1
                    #     save_mode_path = os.path.join(model_path,
                    #                                 'model1_iter_{}_iou_{}.pth'.format(
                    #                                     iter_num, round(best_performance1, 4)))
                    #     save_best = os.path.join(model_path,
                    #                             'model1_best.pth')
                    #     # torch.save(model1.state_dict(), save_mode_path)
                    #     torch.save(model1.state_dict(), save_best)

                    # logging.info(
                    #     'iteration %d : model1_mean_iou : %f' % (iter_num, avg_iou_val1))
                    # logging.info(
                    #     'iteration %d : model1_test_mean_iou : %f' % (iter_num, avg_iou_test1))


                    

                    # if avg_iou_val2 > best_performance2: #保存验证集合上最好的模型
                    #     logging.info("model2"+str(iou_val2))
                    #     best_performance2 = avg_iou_val2
                    #     best_performance_epoch = epoch_num
                    if avg_iou_test2 > best_performance2: #保存测试集合上最好的模型
                        logging.info("model2"+str(iou_test2))
                        best_performance2 = avg_iou_test2
                        best_performance_epoch = epoch_num
                        
                        # 计算并记录达到最佳性能时的总训练耗时
                        best_performance_total_time = time.time() - total_train_start_time
                        logging.info(f"计算并记录达到验证集上达到最优! Epoch: {best_performance_epoch}, 总训练耗时: {best_performance_total_time:.4f}秒")

                        save_mode_path = os.path.join(model_path,
                                                    'model2_iter_{}_mean_iou_{}.pth'.format(
                                                        iter_num, round(best_performance2,4)))
                        save_best = os.path.join(model_path,'model2_best.pth')
                        # torch.save(model2.state_dict(), save_mode_path)
                        torch.save(model2.state_dict(), save_best)
                    # writer.add_scalar('test/model2_mean_iou', avg_iou_val2, epoch_num)
                    writer.add_scalar('test/model2_mean_test_iou', avg_iou_test2, epoch_num)  
                    # logging.info(
                    #     'iteration %d : model2_mean_iou : %f' % (iter_num, avg_iou_val2))
                    logging.info(
                        'iteration %d : model2_mean_test_iou : %f' % (iter_num, avg_iou_test2))
                    model1.train()
                    model2.train()
        writer.add_scalar('info/lr', lr_, epoch_num)
        loss_dist_tensor = torch.tensor(loss_dist)
        writer.add_scalar('info/model1_total_loss', loss_dist_tensor.mean(), epoch_num)
        loss_dist.clear()
        
        supervised_loss_dist_tensor = torch.tensor(supervised_loss_dist)
        writer.add_scalar('info/supervised_loss', supervised_loss_dist_tensor.mean(), epoch_num)
        supervised_loss_dist.clear()
        consistency_loss_dist_tensor = torch.tensor(loss_consistence_dist)
        writer.add_scalar('info/consistency_loss',
                              consistency_loss_dist_tensor.mean(), epoch_num)
        loss_consistence_dist.clear()
        writer.add_scalar('info/consistency_weight',
                              consistency_weight, epoch_num)
        writer.add_scalar('info/pseudo_correction_weight',
                              pseudo_correction_weight, epoch_num)
        writer.add_scalar('info/uncertainty',
                              threshold, epoch_num)
        loss_pse_label_correction_dist_tensor = torch.tensor(loss_pse_label_correction_dist)
        writer.add_scalar('info/pse_label_correction_loss',
                              loss_pse_label_correction_dist_tensor.mean(), epoch_num)
        loss_pse_label_correction_dist.clear()
        
        # Epoch结束时间和耗时计算
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # 记录当前epoch的显存峰值
        current_epoch_peak = torch.cuda.max_memory_cached() / 1024**2  # 转换为MB
        if current_epoch_peak > peak_memory:
            peak_memory = current_epoch_peak
        
        # 打印当前epoch信息并记录到日志
        epoch_info = f"Epoch [{epoch_num+1}/{max_epoch}] - 耗时: {epoch_duration:.4f}秒 - 显存峰值: {current_epoch_peak:.4f}MB - 累计显存峰值: {peak_memory:.4f}MB"
        print(epoch_info)
        logging.info(epoch_info)
    
    # 计算总训练耗时
    total_train_end_time = time.time()
    total_train_duration = total_train_end_time - total_train_start_time
    
    # 打印训练总结并记录到日志
    summary_info = "\n" + "="*60
    print(summary_info)
    logging.info(summary_info)
    
    summary_info = "训练完成!"
    print(summary_info)
    logging.info(summary_info)
    
    summary_info = f"总训练耗时: {total_train_duration:.4f}秒"
    print(summary_info)
    logging.info(summary_info)
    
    summary_info = f"在验证集上达到最佳性能时的Epoch: {best_performance_epoch}, 总训练耗时: {best_performance_total_time:.4f}秒 ({best_performance_total_time/3600:.2f} 小时)"
    print(summary_info)
    logging.info(summary_info)
    
    summary_info = f"最终峰值显存: {peak_memory:.4f}GB"
    print(summary_info)
    logging.info(summary_info)
    
    summary_info = "="*60
    print(summary_info)
    logging.info(summary_info)
    
    writer.close()                     

def main(bili_str,baifenbi):

    snapshot_path=f'/mnt/hys/CPAC_Net/My_mt_with_reg_strong_weak_outputs_75%/20251225_lr_1e-3_500epoch_置信度阈值95e-2_consis_1e-2-11e-2_KL_s_w_threshold_6e-1_4e-1_变形图像也参与训练0.5_aug2_testbest'
    
    if not os.path.exists(snapshot_path):
        print("%s没有，正在创建" % snapshot_path)
        os.makedirs(snapshot_path)
    source1 = '/mnt/hys/CPAC_Net/train_my_mt_with_reg_同个病人_strong_weak_75%.py'
    source2 = '/mnt/hys/CPAC_Net/mt_Dataset.py'
    source3 = '/mnt/hys/CPAC_Net/calzhibiao.py'
    source4 = '/mnt/hys/CPAC_Net/unet_model.py'
    source5 = '/mnt/hys/CPAC_Net/STN_2d.py'
    source6 = '/mnt/hys/CPAC_Net/Attention.py'
    source7 = '/mnt/hys/CPAC_Net/augmentation_2d.py'
    target = snapshot_path+'/'
    shutil.copy(source1,target)
    shutil.copy(source2,target)
    shutil.copy(source3,target)
    shutil.copy(source4,target)
    shutil.copy(source5,target)
    shutil.copy(source6,target)
    shutil.copy(source7,target)
    print("复制成功")
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(snapshot_path+"/log.txt")
    # dir_path=f"/mnt/newdisk/lf/zhibiao/zhibiao_7_25_newdata_uamt_buyongwubiaozhushuju{bili_str}%_lr_1e-2_normalize_new_unbz2"
    dir_path=snapshot_path
    # data_file='data_new_2/train_dict1.pkl'
    # dir_path="zhibiao/zhibiao_9_18_up&down_mgd_newdata_pseudo_lossis_ce"
    if not os.path.exists(dir_path):
        print("%s没有，正在创建" % dir_path)
        os.makedirs(dir_path)
    data_file_list=['/mnt/hys/Datasets/重复测量（hys整理版）/for_5fold_75%/train_dict1.pkl','/mnt/hys/Datasets/重复测量（hys整理版）/for_5fold_75%/train_dict2.pkl','/mnt/hys/Datasets/重复测量（hys整理版）/for_5fold_75%/train_dict3.pkl','/mnt/hys/Datasets/重复测量（hys整理版）/for_5fold_75%/train_dict4.pkl','/mnt/hys/Datasets/重复测量（hys整理版）/for_5fold_75%/train_dict5.pkl']
    wuzhe_zhibiao=[]
    file=open(os.path.join(snapshot_path,'wuzhe.txt'),'w')
    a= 0

    for data_file in data_file_list:
        a+=1
        add_content = "threshold = (0.6+0.4*ramps.sigmoid_rampup(iter_num, max_epoch*len(trainloader))*np.log(2)); T_max 500;consistency_weight = 0.01+0.1*get_current_consistency_weight(epoch_num) CE"
                        
        # add_content = "threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num,max_epoch*len(trainloader)))*np.log(2); T_max 200;consistency_weight = 0.01+0.1*get_current_consistency_weight(epoch_num) CE"
        
        # if(a==1):
        #     continue
        img_path='/mnt/hys/Datasets/重复测量（hys整理版）/img_crop_npy'
        # train(snapshot_path,zhibiao_dir=dir_path,data_file=data_file,baifenbi=baifenbi,fold_num= a)
        test_iou_2=cal_zhibiao_hys_dantongdao_jiandanguiyihua_RCPS(fold_num=a,dir_path=dir_path,img_path=img_path,data_file=data_file,model_path=os.path.join(snapshot_path,data_file.split('/')[-1].split('.')[0],'model2_best.pth'),aux=False,data_type="test",device='gpu',model='unet',add_content=add_content,save_result_img=True)
        wuzhe_zhibiao.append(test_iou_2)
    rows_to_extract = [0, 3, 4]
    # 提取指定行
    extracted_rows = [wuzhe_zhibiao[i] for i in rows_to_extract]    
    # 将提取的行转换为NumPy数组
    extracted_array = np.array(extracted_rows)

    # 计算每个指标的平均值和标准差
    mean_values = np.mean(extracted_array, axis=0)
    std_values = np.std(extracted_array, axis=0)

    print("Mean values:", mean_values)
    print("Standard deviations:", std_values)
    
    # 计算每个指标的平均值和标准差
    print(f"145折交叉验证平均iou:{mean_values[0]},标准差：{std_values[0]}\n,145折交叉验证平均hd95:{mean_values[1]},标准差：{std_values[1]}\n,145折交叉验证平均recall:{mean_values[2]},标准差：{std_values[2]}\n\
        ,145折交叉验证平均sensitivity:{mean_values[3]},标准差：{std_values[3]}\n,145折交叉验证平均dc:{mean_values[4]},标准差：{std_values[4]}\n,145折交叉验证平均jc:{mean_values[5]},标准差：{std_values[5]}\n\
            ,145折交叉验证平均acc:{mean_values[6]},标准差：{std_values[6]}\n")
    wuzhe=np.sum(wuzhe_zhibiao,0)/5
    std = np.std(wuzhe_zhibiao, axis=0)
    
    print(wuzhe)
    print(f"五折交叉验证平均iou:{wuzhe[0]},标准差：{std[0]}\n,五折交叉验证平均hd95:{wuzhe[1]},标准差：{std[1]}\n,五折交叉验证平均recall:{wuzhe[2]},标准差：{std[2]}\n\
        ,五折交叉验证平均sensitivity:{wuzhe[3]},标准差：{std[3]}\n,五折交叉验证平均dc:{wuzhe[4]},标准差：{std[4]}\n,五折交叉验证平均jc:{wuzhe[5]},标准差：{std[5]}\n\
            ,五折交叉验证平均acc:{wuzhe[6]},标准差：{std[6]}\n")
    
    for i in range(0,5):
        file.write(f"{str(wuzhe_zhibiao[i])}\n")
    file.write('\n')
    file.write(str(wuzhe))
    
    file.write('\n')
    file.write(str(std))
    file.write('\n')
    file.write(f"五折交叉验证平均iou:{wuzhe[0]},标准差：{std[0]}\n,五折交叉验证平均hd95:{wuzhe[1]},标准差：{std[1]}\n,五折交叉验证平均recall:{wuzhe[2]},标准差：{std[2]}\n\
        ,五折交叉验证平均sensitivity:{wuzhe[3]},标准差：{std[3]}\n,五折交叉验证平均dc:{wuzhe[4]},标准差：{std[4]}\n,五折交叉验证平均jc:{wuzhe[5]},标准差：{std[5]}\n\
            ,五折交叉验证平均acc:{wuzhe[6]},标准差：{std[6]}\n")
    file.write('\n')
    file.write(f"145折交叉验证平均iou:{mean_values[0]},标准差：{std_values[0]}\n,145折交叉验证平均hd95:{mean_values[1]},标准差：{std_values[1]}\n,145折交叉验证平均recall:{mean_values[2]},标准差：{std_values[2]}\n\
        ,145折交叉验证平均sensitivity:{mean_values[3]},标准差：{std_values[3]}\n,145折交叉验证平均dc:{mean_values[4]},标准差：{std_values[4]}\n,145折交叉验证平均jc:{mean_values[5]},标准差：{std_values[5]}\n\
            ,145折交叉验证平均acc:{mean_values[6]},标准差：{std_values[6]}\n")


def set_random_seed(seed_value):
    """Set the random seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
if __name__ == "__main__":

    set_random_seed(3407)

    main("25",0.75)


