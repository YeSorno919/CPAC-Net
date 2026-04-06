import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from PIL import Image
from torchvision import transforms
# import imgaug.augmenters as iaa
from torch.utils.data import DataLoader
import _pickle as pkl
# import imgaug as ia
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from onehot import *
import edge_utils
import itertools
import torch
from matplotlib import pyplot as plt



def denormalize(tensor, mean, std):
    """
    逆标准化图像。

    参数:
    tensor (torch.Tensor): 标准化的图像张量，形状为 (C, H, W) 或 (B, C, H, W)
    mean (list or tuple): 归一化时使用的均值
    std (list or tuple): 归一化时使用的标准差

    返回:
    torch.Tensor: 恢复到原始像素值范围的图像张量
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # 增加 batch 维度

    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)

    # 逆标准化
    tensor = tensor * std + mean

    # 确保像素值在 [0, 1] 范围内
    tensor = torch.clamp(tensor, 0, 1)

    if tensor.size(0) == 1:
        tensor = tensor.squeeze(0)  # 移除 batch 维度

    return tensor


# import albumentations as al

# class RandomGenerator(object):
#     def __init__(self):
#         self.transform=transforms.Compose([
#     transforms.ToTensor()
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     # ,transforms.Resize((360,740))
# ])
#     def augmentor(self, image):
#         height, width, _ = image.shape
#         sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#         aug = iaa.Sequential([
#             iaa.Fliplr(p=0.5),
#             sometimes(iaa.Affine(
#                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
#                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
#                 rotate=(-20, 20),  # 旋转±45度之间
#                 shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
#                 order=[0, 1]  # 使用最邻近差值或者双线性差值
#                 # cval=(0, 255),  # 全白全黑填充
#                 # mode=ia.ALL  # 定义填充图像外区域的方法
#             )),
#             iaa.SomeOf((0, 5),
#                        [
#                            # 锐化处理
#                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
#                            # 浮雕效果
#                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
#                            # 边缘检测，将检测到的赋值0或者255然后叠在原图上
#                            sometimes(iaa.OneOf([
#                                iaa.EdgeDetect(alpha=(0, 0.7)),
#                                iaa.DirectedEdgeDetect(
#                                    alpha=(0, 0.7), direction=(0.0, 1.0)
#                                ),
#                            ])),
#                            # 将1%到10%的像素设置为黑色
#                            # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖

#                            iaa.CoarseDropout(
#                                (0.03, 0.10), size_percent=(0.01, 0.03),
#                                per_channel=0.2
#                            ),
#                         #    5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
#                            iaa.Invert(0.05, per_channel=True),
#                            # 每个像素随机加减-10到10之间的数
#                            iaa.Add((-10, 10), per_channel=0.5),
#                            # 像素乘上0.5或者1.5之间的数字.
#                            iaa.Multiply((0.5, 1.5), per_channel=0.5),
#                            # 将整个图像的对比度变为原来的一半或者二倍
#                            iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
#                            # 扭曲图像的局部区域
#                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
#                        ],
#                        random_order=True  # 随机的顺序把这些操作用在图像上
#                        )
#                 ])
#         return aug

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         label = SegmentationMapsOnImage(label, shape=image.shape)
#         aug = self.augmentor(image)
#         image, label = aug(image=image, segmentation_maps=label)
#         image = self.transform(image)
#         label = label.get_arr()
#         label = torch.from_numpy(label)
#         sample['image'] = image
#         sample['label'] = label
#         return sample

# transform1 = transforms.Compose([
#     RandomGenerator()
# ])
transform2 = transforms.Compose([
    transforms.ToTensor()
])



# def read_split_data(root: str,save_dir:str, train_rate: float = 0.9):
#     #生成文件
#     # ——————————用数据集路径传递即可———————————#
#     # ——————————生成两个列表 一个为数据集文件名称 另一个为测试集文件名称列表———————————#
#     random.seed(0)  # 保证随机结果可复现
#     assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
#
#     train_images_name = []  # 存储训练集的所有图片名称
#     test_images_name = []  # 存储验证集的所有图片名称
#     images = os.listdir(root)
#     images.sort()
#     # print("images.len",images)
#
#     # 按比例随机采样验证样本
#     train_path = random.sample(images, k=int(len(images) * train_rate))
#     train_file = open(os.path.join(save_dir,"train_data_new.txt"), "w")
#     test_file = open(os.path.join(save_dir,"test_data_new.txt"), "w")
#     for img_name in images:
#         if img_name in train_path:  # 如果该路径在采样的训练集样本中则存入训练集
#             train_images_name.append(img_name)
#             train_file.write(img_name)
#             train_file.write("\n")
#         else:  # 否则存入验证集
#             test_images_name.append(img_name)
#             test_file.write(img_name)
#             test_file.write("\n")
#     train_file.close()
#     test_file.close()
#     print("{} images for training.".format(len(train_images_name)))
#     print("{} images for test.".format(len(test_images_name)))
#     return train_images_name, test_images_name




class BaseDataSets(Dataset):
    def __init__(self,data_file,img_path="data_new_2/image_crop",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
    # def __init__(self,data_file,img_path="data_new_2/image_crop",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor()])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list = train_dict['train_list'][:int(len(train_dict['train_list'])*baifenbi)]#通过文件把数据集变成文件名列表
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list = train_dict['train_list'][int(len(train_dict['train_list'])*baifenbi):]#通过文件把数据集变成文件名列表
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list = train_dict['val_list']
        elif self.split == 'test':
            self.sample_list = train_dict['test_list']

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, case)
        # print(imgA_path)
        imgB_path='data_new_2/label_crop/%s.gif' % name
        img = cv2.imread(imgA_path)
        # print(img.shape)
        # img= img.resize((704, 320))
        label = Image.open(imgB_path)
        # img = cv2.resize(img, (224, 224))  # resize里的改变后大小是w*h .size是h*w
        # label= label.resize((224,224))
        # img_xia=cv2.imread(os.path.join("data_new_2/img_xia",case))
        img=cv2.resize(img,(704, 320))
        # img_xia=al.VerticalFlip(p=0.5)(image=img_xia)['image']
        label= label.resize((704, 320))

        label = np.array(label)
        _edgemap = label.copy()
        _edgemap = edge_utils.mask_to_onehot(_edgemap, 2)
        _edgemap = edge_utils.onehot_to_binary_edges(
            _edgemap, 2, 2
        )  # This needs fixing
        edgemap = torch.from_numpy(_edgemap).float()
        # data={"image":img,"mask":label}
        # Ver=al.VerticalFlip(p=0.5)(**data)
        # img,label=Ver['image'],Ver['mask']
        # label_onehot=onehot(label,2)
        # label_onehot = label_onehot.transpose(2, 0, 1)
        # label_onehot=label_onehot.astype('uint8')
        # label_onehot[0] = cv2.distanceTransform(label_onehot[0], cv2.DIST_L1, 3)
        # label_onehot[1] = cv2.distanceTransform(label_onehot[1], cv2.DIST_L1, 3)
        # sample = {'image': img, 'label': label,"img_xia":img_xia,"label_dist":label_onehot}
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"edgemap":edgemap,"label_x2":cv2.resize(label,(352,160)),"label_x3":cv2.resize(label,(176,80)),"label_x4":cv2.resize(label,(88,40)),"label_x5":cv2.resize(label,(44,20))}
            # sample = {'image': img, "label": label,"label_x1":cv2.resize(label,(704,320)),"label_x2":cv2.resize(label,(352,160)),"label_x3":cv2.resize(label,(176,80)),"label_x4":cv2.resize(label,(88,40)),"label_x5":cv2.resize(label,(44,20)),"name":name}


        # "label_x3":cv2.resize(label,(176,80))}
        # 
        # "img_xia":img_xia,"label_x3":cv2.resize(label,(176,80))}
            # "label_x3":cv2.resize(label,(176,80)),\
            # "label_x2":cv2.resize(label,(352,160))
            # }
                # "label_x5":cv2.resize(label,(44,20))}
        # if self.split == "train":
        #     sample = self.transform1(sample)
        # else:
            # print("fore:")
            # print(sample['image'].shape)
        sample['image']=self.transform2(sample['image'])#不用aug
        # sample['img_xia']=self.transform2(img_xia)
            # print("hou:")
            # print(sample['image'].shape)
            # print(sample['image'].shape)
        sample["idx"] = case
        return sample
    
class MGD_BaseDataSets(Dataset):
    def __init__(self,data_file,img_path="Expore MGD1k Dataset/Original Images",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list = train_dict['train_list'][:int(len(train_dict['train_list'])*baifenbi)]#通过文件把数据集变成文件名列表
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list = train_dict['train_list'][int(len(train_dict['train_list'])*baifenbi):]#通过文件把数据集变成文件名列表
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list = train_dict['val_list']
        elif self.split == 'test':
            self.sample_list = train_dict['test_list']

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, case.replace(".png", ".JPG"))
        imgB_path='Expore MGD1k Dataset/Meibomian Gland Labels/%s.png' % name
        img = cv2.imread(imgA_path,0)
        # print(img.shape)
        # img= img.resize((704, 320))
        label = Image.open(imgB_path)
        # img = cv2.resize(img, (224, 224))  # resize里的改变后大小是w*h .size是h*w
        # label= label.resize((224,224))
        # img_xia=cv2.imread(os.path.join("data_new_2/img_xia",case))
        img=cv2.resize(img,(640,320))[:,:,None]
        # img_xia=al.VerticalFlip(p=0.5)(image=img_xia)['image']
        label= label.resize((640, 320))

        label = np.array(label)//255
        # _edgemap = label.copy()
        # _edgemap = edge_utils.mask_to_onehot(_edgemap, 2)
        # _edgemap = edge_utils.onehot_to_binary_edges(
        #     _edgemap, 2, 2
        # )  # This needs fixing
        # edgemap = torch.from_numpy(_edgemap).float()
        # data={"image":img,"mask":label}
        # Ver=al.VerticalFlip(p=0.5)(**data)
        # img,label=Ver['image'],Ver['mask']
        # label_onehot=onehot(label,2)
        # label_onehot = label_onehot.transpose(2, 0, 1)
        # label_onehot=label_onehot.astype('uint8')
        # label_onehot[0] = cv2.distanceTransform(label_onehot[0], cv2.DIST_L1, 3)
        # label_onehot[1] = cv2.distanceTransform(label_onehot[1], cv2.DIST_L1, 3)
        # sample = {'image': img, 'label': label,"img_xia":img_xia,"label_dist":label_onehot}
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}
        # "label_x3":cv2.resize(label,(176,80))}
        # 
        # "img_xia":img_xia,"label_x3":cv2.resize(label,(176,80))}
            # "label_x3":cv2.resize(label,(176,80)),\
            # "label_x2":cv2.resize(label,(352,160))
            # }
                # "label_x5":cv2.resize(label,(44,20))}
        # if self.split == "train":
        #     sample = self.transform1(sample)
        # else:
            # print("fore:")
            # print(sample['image'].shape)
        # print(sample['image'].shape)
        sample['image']=self.transform2(sample['image'])#不用aug
        # sample['img_xia']=self.transform2(img_xia)
            # print("hou:")
            # print(sample['image'].shape)
            # print(sample['image'].shape)
        sample["idx"] = case
        return sample



# class unlabelDataSets(Dataset):
#     def __init__(self,data_file,img_path="data_new_2/image_crop",zhibiao_dir=None, split='train', num=None, transform1 = None,transform2=transforms.Compose([transforms.ToTensor()])):
#         self.sample_list = []
#         self.split = split
#         self.transform1 = transform1
#         self.transform2 = transform2
#         self.img_path=img_path
#         self.zhibiao_dir=zhibiao_dir
#         file = open(data_file, 'rb')
#         train_dict = pkl.load(file)
#         if self.split == 'train':
#             self.sample_list = train_dict['train_list'][len(train_dict['train_list'])//2:]#通过文件把数据集变成文件名列表
#             print(self.sample_list)
#         elif self.split == 'val':
#             self.sample_list = train_dict['val_list']
#         elif self.split == 'test':
#             self.sample_list = train_dict['test_list']

#         print("total {} samples".format(len(self.sample_list)))

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         case = self.sample_list[idx]#取出名字
#         name = case.split(".")[0]
#         imgA_path=os.path.join(self.img_path, case)
#         img = cv2.imread(imgA_path)
#         img = cv2.resize(img, (704, 320))  # resize里的改变后大小是w*h .size是h*w
#         img_xia=cv2.imread(os.path.join("data_new_2/img_xia",case))
#         img_xia=cv2.resize(img_xia,(704,320))
#         # img_xia=al.VerticalFlip(p=0.5)(image=img_xia)['image']
#         data={"image":img}
#         Ver=al.VerticalFlip(p=0.5)(**data)
#         # img,label=Ver['image'],Ver['mask']

#         # sample = {'image': img, 'label': label,"img_xia":img_xia,"label_dist":label_onehot}
#         sample = {'image': img}
#         # "label_x3":cv2.resize(label,(176,80))}
#         # 
#         # "img_xia":img_xia,"label_x3":cv2.resize(label,(176,80))}
#             # "label_x3":cv2.resize(label,(176,80)),\
#             # "label_x2":cv2.resize(label,(352,160))
#             # }
#                 # "label_x5":cv2.resize(label,(44,20))}
#         # if self.split == "train":
#         #     sample = self.transform1(sample)
#         # else:
#             # print("fore:")
#             # print(sample['image'].shape)
#         sample['image']=transform2(sample['image'])#不用aug
#         sample['img_xia']=transform2(img_xia)
#             # print("hou:")
#             # print(sample['image'].shape)
#             # print(sample['image'].shape)
#         sample["idx"] = name
#         return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# 三通道 特殊标准化
class hysDataSets(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a["Altas"][0] for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list=[b for a in train_dict['train'] for b in a['OtherImg'] ]
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)
        # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
        img = np.tile(img[:, :, None], (1, 1, 3))
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path)
        label = np.where(label<=0,0,label)
        label = np.where(label>0,1,label)
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

        sample['image']=self.transform2(sample['image'])#不用aug  

        sample["idx"] = case
        return sample
# 三通道 同个病人特殊标准化
class hysDataSets_fenbingren(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = case['Altas'][0].split(".")[0]
            unlabelimg_list = random.sample(case["OtherImg"], 1)
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = np.tile(img[:, :, None], (1, 1, 3))
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            unlabelimg = np.tile(unlabelimg[:, :, None], (1, 1, 3))
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40)),'unlabel_image':unlabelimg}
            sample['image']=self.transform2(sample['image'])
            sample['unlabel_image']=self.transform2(sample['unlabel_image'])
            sample["idx"] = case
        else:
            case = self.sample_list[idx]#取出名字
            name = case.split(".")[0]
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = np.tile(img[:, :, None], (1, 1, 3))
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

            sample['image']=self.transform2(sample['image'])#不用aug  

            sample["idx"] = case
            return sample
# 单通道 特殊标准化 分病人
class hysDataSets_bingren_pingguweibiaoqianzhiliang(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = case['Altas'][0].split(".")[0]
            unlabelimg_list = random.sample(case["OtherImg"], 1)
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = img[:,:,None]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            
            unlabellab_path=os.path.join(parent_dir,'seg_npy', unlabelname+'.npy')
            unlabellab = np.load(unlabellab_path)
            unlabellab = np.where(unlabellab<=0,0,unlabellab)
            unlabellab = np.where(unlabellab>0,1,unlabellab)
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40)),'unlabel_image':unlabelimg,'unlabel_label':unlabellab}
            sample['image']=self.transform2(sample['image'])
            sample['unlabel_image']=self.transform2(sample['unlabel_image'])
            sample["idx"] = case
        else:
            case = self.sample_list[idx]#病人的字典
            name = case.split(".")[0]    
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = img[:,:,None]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

            sample['image']=self.transform2(sample['image'])#不用aug  

            sample["idx"] = case
        return sample
class hysDataSets_bingren(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = case['Altas'][0].split(".")[0]
            unlabelimg_list = random.sample(case["OtherImg"], 1)
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = img[:,:,None]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40)),'unlabel_image':unlabelimg}
            sample['image']=self.transform2(sample['image'])
            sample['unlabel_image']=self.transform2(sample['unlabel_image'])
            sample["idx"] = case
        else:
            case = self.sample_list[idx]#病人的字典
            name = case.split(".")[0]    
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = img[:,:,None]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

            sample['image']=self.transform2(sample['image'])#不用aug  

            sample["idx"] = case
        return sample
class hysDataSets_bingren_50_75percent(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = random.sample(case['Altas'], 1)[0].split(".")[0]
            unlabelimg_list = random.sample(case["OtherImg"], 1)
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = img[:,:,None]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40)),'unlabel_image':unlabelimg}
            sample['image']=self.transform2(sample['image'])
            sample['unlabel_image']=self.transform2(sample['unlabel_image'])
            sample["idx"] = case
        else:
            case = self.sample_list[idx]#病人的字典
            name = case.split(".")[0]    
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = img[:,:,None]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

            sample['image']=self.transform2(sample['image'])#不用aug  

            sample["idx"] = case
        return sample
    
# 单通道 分病人，简单归一化
class hysDataSets_dantongdao_jiandanguiyihua(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,with_affine_field=False):
        self.with_affine_field=with_affine_field
        self.sample_list = []
        self.split = split
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)
    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int') 
        input_shape = y.shape
        # 检查最后一个维度是否为1，以及是否是多维度，如果满足条件则将最后一个维度去掉
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        # print("intputshape",input_shape)
        # 将y展平成一维数组，其大小为样本数量
        y = y.ravel()
        # 如类别未指定，则通过最大的类别标签号加1。即加上背景这一类
        if not num_classes:
            num_classes = np.max(y) + 1
        #n为像素的個數
        n = y.shape[0]
        # print("像素個數=",n)
        # 创建一个全0矩阵,用于存储独热编码
        categorical = np.zeros((num_classes, n))
        # print(categorical.shape)
        # 每一行表示一个类别,每一列代表一个样本.对应位置为1则表示该样本属于该类别
        # print(np.max(y))
        categorical[y, np.arange(n)] = 1
        # 将设置输出的大小. 若input_shape= (2,2),则output_shape=(num_classes,2,2)
        output_shape = (num_classes,) + input_shape
        # 将数组重新塑造设置的输出大小
        categorical = np.reshape(categorical, output_shape) #(3,350,740)
        # print(categorical.shape)
        return categorical
    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = case['Altas'][0].split(".")[0]
            unlabelimg_list = random.sample(case["OtherImg"], 1)
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            
            img = img/255.
            img = img.astype(np.float32)
            img = img[np.newaxis, :, :]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            unlabelimg = unlabelimg/255.
            unlabelimg = unlabelimg.astype(np.float32)
            unlabelimg = unlabelimg[np.newaxis, :, :]

            sample = {'image': img, "label": label,'unlabel_image':unlabelimg}
            sample["labelimg_name"] = labelimg_name
            sample["unlabelname"] = unlabelname
            # 读取无标签图像的标签是对比伪标签的生成效果。
            if(True):
                unlabelimg_label_path = os.path.join(parent_dir,'seg_npy',unlabelname+'.npy')
                unlabelimg_label = np.load(unlabelimg_label_path)
                unlabelimg_label = np.where(unlabelimg_label<=0,0,unlabelimg_label)
                unlabelimg_label = np.where(unlabelimg_label>0,1,unlabelimg_label)
                sample["unlabel_img_label_for_compair"] = unlabelimg_label
            if(self.with_affine_field):
                labelimg_name_to_unlabelname = labelimg_name+"to"+unlabelname 
                l_u_img = np.load(os.path.join(parent_dir, 'all_pair_affine_img_npy', labelimg_name_to_unlabelname+".npy"))
                l_u_img = l_u_img/255.
                l_u_img = l_u_img.astype(np.float32)
                l_u_img = l_u_img[np.newaxis, :, :]
                l_u_seg=np.load(os.path.join(parent_dir, 'all_pair_affine_seg_npy', labelimg_name_to_unlabelname+".npy"))
                l_u_seg = np.where(l_u_seg<=0,0,l_u_seg)
                l_u_seg = np.where(l_u_seg>0,1,l_u_seg)
                l_u_seg_onehot = self.to_categorical(l_u_seg, 2)
                l_u_seg_onehot = l_u_seg_onehot.astype(np.float32)
                sample["l_u_img"] = l_u_img
                sample["l_u_seg"] = l_u_seg
                sample["l_u_seg_onehot"] = l_u_seg_onehot
                
        else:
            case = self.sample_list[idx]#病人的字典
            name = case.split(".")[0]    
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            img = img/255.
            img = img.astype(np.float32)
            img = img[np.newaxis, :, :]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label}

            sample["idx"] = case
        return sample    

# 单通道 分病人，简单归一化 50%的标注比例
class hysDataSets_dantongdao_jiandanguiyihua_50_75percent(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,with_affine_field=False):
        self.with_affine_field=with_affine_field
        self.sample_list = []
        self.split = split
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)
    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int') 
        input_shape = y.shape
        # 检查最后一个维度是否为1，以及是否是多维度，如果满足条件则将最后一个维度去掉
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        # print("intputshape",input_shape)
        # 将y展平成一维数组，其大小为样本数量
        y = y.ravel()
        # 如类别未指定，则通过最大的类别标签号加1。即加上背景这一类
        if not num_classes:
            num_classes = np.max(y) + 1
        #n为像素的個數
        n = y.shape[0]
        # print("像素個數=",n)
        # 创建一个全0矩阵,用于存储独热编码
        categorical = np.zeros((num_classes, n))
        # print(categorical.shape)
        # 每一行表示一个类别,每一列代表一个样本.对应位置为1则表示该样本属于该类别
        # print(np.max(y))
        categorical[y, np.arange(n)] = 1
        # 将设置输出的大小. 若input_shape= (2,2),则output_shape=(num_classes,2,2)
        output_shape = (num_classes,) + input_shape
        # 将数组重新塑造设置的输出大小
        categorical = np.reshape(categorical, output_shape) #(3,350,740)
        # print(categorical.shape)
        return categorical
    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = random.sample(case["Altas"], 1)[0]
            z = case["Altas"].copy()
            z.remove(labelimg_name)
            labelimg_name = labelimg_name.split(".")[0]
            t =z+case["OtherImg"]
            unlabelimg_list = random.sample(t, 1)
            
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            
            img = img/255.
            img = img.astype(np.float32)
            img = img[np.newaxis, :, :]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            unlabelimg = unlabelimg/255.
            unlabelimg = unlabelimg.astype(np.float32)
            unlabelimg = unlabelimg[np.newaxis, :, :]

            sample = {'image': img, "label": label,'unlabel_image':unlabelimg}
            sample["labelimg_name"] = labelimg_name
            sample["unlabelname"] = unlabelname
            # 读取无标签图像的标签是对比伪标签的生成效果。
            if(True):
                unlabelimg_label_path = os.path.join(parent_dir,'seg_npy',unlabelname+'.npy')
                unlabelimg_label = np.load(unlabelimg_label_path)
                unlabelimg_label = np.where(unlabelimg_label<=0,0,unlabelimg_label)
                unlabelimg_label = np.where(unlabelimg_label>0,1,unlabelimg_label)
                sample["unlabel_img_label_for_compair"] = unlabelimg_label
            if(self.with_affine_field):
                labelimg_name_to_unlabelname = labelimg_name+"to"+unlabelname 
                l_u_img = np.load(os.path.join(parent_dir, 'all_pair_affine_img_npy', labelimg_name_to_unlabelname+".npy"))
                l_u_img = l_u_img/255.
                l_u_img = l_u_img.astype(np.float32)
                l_u_img = l_u_img[np.newaxis, :, :]
                l_u_seg=np.load(os.path.join(parent_dir, 'all_pair_affine_seg_npy', labelimg_name_to_unlabelname+".npy"))
                l_u_seg = np.where(l_u_seg<=0,0,l_u_seg)
                l_u_seg = np.where(l_u_seg>0,1,l_u_seg)
                l_u_seg_onehot = self.to_categorical(l_u_seg, 2)
                l_u_seg_onehot = l_u_seg_onehot.astype(np.float32)
                sample["l_u_img"] = l_u_img
                sample["l_u_seg"] = l_u_seg
                sample["l_u_seg_onehot"] = l_u_seg_onehot
                
        else:
            case = self.sample_list[idx]#病人的字典
            name = case.split(".")[0]    
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            img = img/255.
            img = img.astype(np.float32)
            img = img[np.newaxis, :, :]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label}

            sample["idx"] = case
        return sample    
 
 # 单通道 分病人，简单归一化 33%的标注量，4张图有一张用于测试见过病人的分割效果。
class hysDataSets_dantongdao_jiandanguiyihua_33percent(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,with_affine_field=False):
        self.with_affine_field=with_affine_field
        self.sample_list = []
        self.seen_but_no_train_num  = 0 
        self.seen_but_no_train_list = []
        self.split = split
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            self.seen_but_no_train_num  = len(self.sample_list)
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            # self.sample_list = [a['OtherImg'][-1] for a in train_dict['train'] ] #记录见过的病人但未参与训练的病人图像
            t = [b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list= self.sample_list+t+z
        elif self.split == 'test':
            self.sample_list = [a['OtherImg'][-1] for a in train_dict['train'] ] #记录见过的病人但未参与训练的病人图像
            self.seen_but_no_train_num  = len(self.sample_list)
            self.seen_but_no_train_list = self.sample_list.copy()
            t = [b for a in train_dict['test'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['test']]
            self.sample_list= self.sample_list+t+z
        print("total {} samples".format(len(self.sample_list)))
    def __len__(self):
        return len(self.sample_list)
    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int') 
        input_shape = y.shape
        # 检查最后一个维度是否为1，以及是否是多维度，如果满足条件则将最后一个维度去掉
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        # print("intputshape",input_shape)
        # 将y展平成一维数组，其大小为样本数量
        y = y.ravel()
        # 如类别未指定，则通过最大的类别标签号加1。即加上背景这一类
        if not num_classes:
            num_classes = np.max(y) + 1
        #n为像素的個數
        n = y.shape[0]
        # print("像素個數=",n)
        # 创建一个全0矩阵,用于存储独热编码
        categorical = np.zeros((num_classes, n))
        # print(categorical.shape)
        # 每一行表示一个类别,每一列代表一个样本.对应位置为1则表示该样本属于该类别
        # print(np.max(y))
        categorical[y, np.arange(n)] = 1
        # 将设置输出的大小. 若input_shape= (2,2),则output_shape=(num_classes,2,2)
        output_shape = (num_classes,) + input_shape
        # 将数组重新塑造设置的输出大小
        categorical = np.reshape(categorical, output_shape) #(3,350,740)
        # print(categorical.shape)
        return categorical
    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = case['Altas'][0].split(".")[0]
            unlabeltotal = case["OtherImg"][:2] #取前2个作为无标签数据的来源。
            unlabelimg_list = random.sample(unlabeltotal, 1)
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            
            img = img/255.
            img = img.astype(np.float32)
            img = img[np.newaxis, :, :]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            unlabelimg = unlabelimg/255.
            unlabelimg = unlabelimg.astype(np.float32)
            unlabelimg = unlabelimg[np.newaxis, :, :]

            sample = {'image': img, "label": label,'unlabel_image':unlabelimg}
            sample["labelimg_name"] = labelimg_name
            sample["unlabelname"] = unlabelname
            # 读取无标签图像的标签是对比伪标签的生成效果。
            if(False):
                unlabelimg_label_path = os.path.join(parent_dir,'seg_npy',unlabelname+'.npy')
                unlabelimg_label = np.load(unlabelimg_label_path)
                unlabelimg_label = np.where(unlabelimg_label<=0,0,unlabelimg_label)
                unlabelimg_label = np.where(unlabelimg_label>0,1,unlabelimg_label)
                sample["unlabel_img_label_for_compair"] = unlabelimg_label
            if(self.with_affine_field):
                labelimg_name_to_unlabelname = labelimg_name+"to"+unlabelname 
                l_u_img = np.load(os.path.join(parent_dir, 'all_pair_affine_img_npy', labelimg_name_to_unlabelname+".npy"))
                l_u_img = l_u_img/255.
                l_u_img = l_u_img.astype(np.float32)
                l_u_img = l_u_img[np.newaxis, :, :]
                l_u_seg=np.load(os.path.join(parent_dir, 'all_pair_affine_seg_npy', labelimg_name_to_unlabelname+".npy"))
                l_u_seg = np.where(l_u_seg<=0,0,l_u_seg)
                l_u_seg = np.where(l_u_seg>0,1,l_u_seg)
                l_u_seg_onehot = self.to_categorical(l_u_seg, 2)
                l_u_seg_onehot = l_u_seg_onehot.astype(np.float32)
                sample["l_u_img"] = l_u_img
                sample["l_u_seg"] = l_u_seg
                sample["l_u_seg_onehot"] = l_u_seg_onehot
                
        else:
            case = self.sample_list[idx]#病人的字典
            name = case.split(".")[0]    
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            img = img/255.
            img = img.astype(np.float32)
            img = img[np.newaxis, :, :]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label}

            sample["idx"] = case
            sample['seen_but_no_train_num']= self.seen_but_no_train_num
            
            return sample
            
        return sample    
 
# 单通道 分病人，简单归一化
class hysDataSets_dantongdao_jiandanguiyihua_with_roi(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,with_affine_field=False):
        self.with_affine_field=with_affine_field
        self.sample_list = []
        self.split = split
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)
    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int') 
        input_shape = y.shape
        # 检查最后一个维度是否为1，以及是否是多维度，如果满足条件则将最后一个维度去掉
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        # print("intputshape",input_shape)
        # 将y展平成一维数组，其大小为样本数量
        y = y.ravel()
        # 如类别未指定，则通过最大的类别标签号加1。即加上背景这一类
        if not num_classes:
            num_classes = np.max(y) + 1
        #n为像素的個數
        n = y.shape[0]
        # print("像素個數=",n)
        # 创建一个全0矩阵,用于存储独热编码
        categorical = np.zeros((num_classes, n))
        # print(categorical.shape)
        # 每一行表示一个类别,每一列代表一个样本.对应位置为1则表示该样本属于该类别
        # print(np.max(y))
        categorical[y, np.arange(n)] = 1
        # 将设置输出的大小. 若input_shape= (2,2),则output_shape=(num_classes,2,2)
        output_shape = (num_classes,) + input_shape
        # 将数组重新塑造设置的输出大小
        categorical = np.reshape(categorical, output_shape) #(3,350,740)
        # print(categorical.shape)
        return categorical
    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = case['Altas'][0].split(".")[0]
            unlabelimg_list = random.sample(case["OtherImg"], 1)
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            
            img = img/255.
            img = img.astype(np.float32)
            img = img[np.newaxis, :, :]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            
            roi_path = os.path.join(parent_dir,'roi_npy',labelimg_name+'.npy')
            roi = np.load(roi_path)
            roi = np.where(roi<=0,0,roi)
            roi = np.where(roi>0,1,roi)
            
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            unlabelimg = unlabelimg/255.
            unlabelimg = unlabelimg.astype(np.float32)
            unlabelimg = unlabelimg[np.newaxis, :, :]
            sample = {'image': img, "label": label,'unlabel_image':unlabelimg,'roi':roi}
            sample["labelimg_name"] = labelimg_name
            sample["unlabelname"] = unlabelname
            if(self.with_affine_field):
                labelimg_name_to_unlabelname = labelimg_name+"to"+unlabelname 
                l_u_img = np.load(os.path.join(parent_dir, 'all_pair_affine_img_npy', labelimg_name_to_unlabelname+".npy"))
                l_u_img = l_u_img/255.
                l_u_img = l_u_img.astype(np.float32)
                l_u_img = l_u_img[np.newaxis, :, :]
                l_u_seg=np.load(os.path.join(parent_dir, 'all_pair_affine_seg_npy', labelimg_name_to_unlabelname+".npy"))
                l_u_seg = np.where(l_u_seg<=0,0,l_u_seg)
                l_u_seg = np.where(l_u_seg>0,1,l_u_seg)
                l_u_seg_onehot = self.to_categorical(l_u_seg, 2)
                l_u_seg_onehot = l_u_seg_onehot.astype(np.float32)
                sample["l_u_img"] = l_u_img
                sample["l_u_seg"] = l_u_seg
                sample["l_u_seg_onehot"] = l_u_seg_onehot
                
        else:
            case = self.sample_list[idx]#病人的字典
            name = case.split(".")[0]    
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            img = img/255.
            img = img.astype(np.float32)
            img = img[np.newaxis, :, :]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label = np.where(label<=0,0,label)
            label = np.where(label>0,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label}

            sample["idx"] = case
        return sample    
    
# 单通道 特殊标准化 分病人
class hysDataSets_bingren_resizeto672(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#病人的字典
        if self.split == 'train':
            labelimg_name = case['Altas'][0].split(".")[0]
            unlabelimg_list = random.sample(case["OtherImg"], 1)
            imgA_path=os.path.join(self.img_path, labelimg_name+'.npy')
            img = np.load(imgA_path)
            img = cv2.resize(img,(672,672))
            # 扩展为3通道的,后面的self.transform2会将其进行归一化并转成3,350,740
            img = img[:,:,None]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',labelimg_name+'.npy')
            label = np.load(imgB_path)
            label =  cv2.resize(label, (672, 672))
            label = np.where(label<=127.5,0,label)
            label = np.where(label>127.5,1,label)
            unlabelname = unlabelimg_list[0].split(".")[0]
            unlabelimg_path=os.path.join(self.img_path, unlabelname+'.npy')
            unlabelimg = np.load(unlabelimg_path)
            unlabelimg = cv2.resize(unlabelimg,(672,672))
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40)),'unlabel_image':unlabelimg}
            sample['image']=self.transform2(sample['image'])
            sample['unlabel_image']=self.transform2(sample['unlabel_image'])
            sample["idx"] = case
        else:
            case = self.sample_list[idx]#病人的字典
            name = case.split(".")[0]    
            imgA_path=os.path.join(self.img_path, name+'.npy')
            img = np.load(imgA_path)
            img = cv2.resize(img,(672,672))
            img = img[:,:,None]
            parent_dir = os.path.dirname(self.img_path)
            imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
            label = np.load(imgB_path)
            label =  cv2.resize(label, (672, 672))
            label = np.where(label<=127.5,0,label)
            label = np.where(label>127.5,1,label)
            if self.split=="unlabel_train":
                sample = {'image': img}
            else:
                sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

            sample['image']=self.transform2(sample['image'])#不用aug  

            sample["idx"] = case
        return sample
# 单通道 简单标准化
class hysDataSetsdantongdao(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
        self.sample_list = []
        self.split = split
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a["Altas"][0] for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list=[b for a in train_dict['train'] for b in a['OtherImg'] ]
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)

        img = img[None,:, :]
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path)
        label = np.where(label<=0,0,label)
        label = np.where(label>0,1,label)
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}


        sample['image'] = sample['image'] / 255.
        sample['image'] = sample['image'].astype(np.float32)

        sample['image'] = torch.from_numpy(sample['image'])
        sample["idx"] = case
        return sample
    
# 单通道 简单归一化 全监督
class hysDataSetsdantongdao_quanjiandu(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
        self.sample_list = []
        self.split = split
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a["Altas"][0] for a in train_dict['train']]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
            print("标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)

        img = img[None,:, :]
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path)
        label = np.where(label<=0,0,label)
        label = np.where(label>0,1,label)

        sample = {'image': img, "label": label}


        sample['image'] = sample['image'] / 255.
        sample['image'] = sample['image'].astype(np.float32)

        sample['image'] = torch.from_numpy(sample['image'])
        sample["idx"] = case
        return sample
    
   

# 单通道 特殊标准化
class hysDataSetsdantongdaoteshubiaozhunhua(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
    # def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.1458], std=[0.2480])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            for a in train_dict['train']:
                self.sample_list+=a["Altas"]  
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list=[b for a in train_dict['train'] for b in a['OtherImg'] ]
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            for a in train_dict['val']:
                self.sample_list+=a["Altas"]  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            for a in train_dict['test']:
                self.sample_list+=a["Altas"]  

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)

        img = img[:, :, None]
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path)
        label = np.where(label<=0,0,label)
        label = np.where(label>0,1,label)
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

        sample['image']=self.transform2(sample['image'])#不用aug  

        sample["idx"] = case
        return sample
# 单通道 特殊标准化
class hysDataSetsdantongdaoteshubiaozhunhua_resize(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
    # def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.1458], std=[0.2480])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            for a in train_dict['train']:
                self.sample_list+=a["Altas"]  
            print("标签数据为",self.sample_list)
            print("标签数据长度为",len(self.sample_list))
        elif self.split == 'unlabel_train':
            self.sample_list=[b for a in train_dict['train'] for b in a['OtherImg'] ]
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            for a in train_dict['val']:
                self.sample_list+=a["Altas"]  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            for a in train_dict['test']:
                self.sample_list+=a["Altas"]  

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)
        img=img.T
        original_shape = (740, 350)
        new_shape = (752, 352)
        
        # 创建一个新的全0数组，形状为新的形状
        resized_image_data = np.zeros(new_shape, dtype=img.dtype)
        
        # 计算原数组在新数组中的起始位置（通常是左上角）
        start_row, start_col = 0, 0
        
        # 计算可以复制的区域大小（不超过原数组和新数组的边界）
        end_row = min(original_shape[0], new_shape[0])
        end_col = min(original_shape[1], new_shape[1])
        
        # 将原数组的数据复制到新数组的适当位置
        resized_image_data[start_row:end_row, start_col:end_col] = img[:end_row, :end_col]
 
        img = resized_image_data[:, :, None]
        
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path).T
        resized_label = np.zeros(new_shape, dtype=label.dtype)
        resized_label[start_row:end_row, start_col:end_col] = label[:end_row, :end_col]
        label = resized_label
        label = np.where(label<=0,0,label)
        label = np.where(label>0,1,label)
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(94,44)).T}

        sample['image']=self.transform2(sample['image'])#不用aug  

        sample["idx"] = case
        return sample

# 单通道 特殊标准化
class hysDataSetsdantongdaoteshubiaozhunhua_bingguweibiaoqianzhiliang(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
    # def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.1458], std=[0.2480])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a["Altas"][0] for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list=[b for a in train_dict['train'] for b in a['OtherImg'] ]
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)

        img = img[:, :, None]
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path)
        label = np.where(label<=0,0,label)
        label = np.where(label>0,1,label)
        if self.split=="unlabel_train":
            
            sample = {'image': img,"unlabel_label": label}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

        sample['image']=self.transform2(sample['image'])#不用aug  

        sample["idx"] = case
        return sample



class hysDataSetsdantongdaoteshubiaozhunhua_resizeto672(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
    # def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.1458], std=[0.2480])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a["Altas"][0] for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list=[b for a in train_dict['train'] for b in a['OtherImg'] ]
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)
        img = cv2.resize(img, (672, 672))  # resize里的改变后大小是w*h .size是h*w

        img = img[:, :, None]
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path)
        # plt.imshow(label,cmap='gray')
        # plt.savefig(os.path.join('/mnt/hys/lf_codes','zzzresizebefore2.jpg'))
        label =  cv2.resize(label, (672, 672))
        # labelresize = label
        # plt.imshow(labelresize,cmap='gray')
        # plt.savefig(os.path.join('/mnt/hys/lf_codes','zzzzzafter2.jpg'))
        label = np.where(label<=127.5,0,label)
        label = np.where(label>127.5,1,label)
        # er=label
        # plt.imshow(er,cmap='gray')
        # plt.savefig(os.path.join('/mnt/hys/lf_codes','zzzzzlabel2.jpg'))
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

        sample['image']=self.transform2(sample['image'])#不用aug  

        sample["idx"] = case
        return sample
class hysDataSetsdantongdaoteshubiaozhunhua_resizeto672_Xpercent(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=[0.229])])):
    # def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.1458], std=[0.2480])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            for a in train_dict["train"]:
                self.sample_list+=a["Altas"]
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list=[b for a in train_dict['train'] for b in a['OtherImg'] ]
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            for a in train_dict['val']:
                self.sample_list+=a["Altas"]  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            for a in train_dict['test']:
                self.sample_list+=a["Altas"]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)
        img = cv2.resize(img, (672, 672))  # resize里的改变后大小是w*h .size是h*w

        img = img[:, :, None]
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path)
        # plt.imshow(label,cmap='gray')
        # plt.savefig(os.path.join('/mnt/hys/lf_codes','zzzresizebefore2.jpg'))
        label =  cv2.resize(label, (672, 672))
        # labelresize = label
        # plt.imshow(labelresize,cmap='gray')
        # plt.savefig(os.path.join('/mnt/hys/lf_codes','zzzzzafter2.jpg'))
        label = np.where(label<=127.5,0,label)
        label = np.where(label>127.5,1,label)
        # er=label
        # plt.imshow(er,cmap='gray')
        # plt.savefig(os.path.join('/mnt/hys/lf_codes','zzzzzlabel2.jpg'))
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

        sample['image']=self.transform2(sample['image'])#不用aug  

        sample["idx"] = case
        return sample

 # 三通道 简单标准化   
class hysDataSets3tongdaojiandanguiyihua(Dataset):
    def __init__(self,data_file,img_path="",zhibiao_dir=None, split='train',baifenbi=1,transform1 = None,transform2=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])):
        self.sample_list = []
        self.split = split
        self.transform1 = transform1
        self.transform2 = transform2
        self.img_path=img_path
        self.zhibiao_dir=zhibiao_dir
        file = open(data_file, 'rb')
        train_dict = pkl.load(file)
        if self.split == 'train':
            self.sample_list=[a["Altas"][0] for a in train_dict['train']]
            print("标签数据为",self.sample_list)
        elif self.split == 'unlabel_train':
            self.sample_list=[b for a in train_dict['train'] for b in a['OtherImg'] ]
            print("无标签数据为",self.sample_list)
        elif self.split == 'val':
            self.sample_list=[b for a in train_dict['val'] for b in a['OtherImg'] ]
            z = [a["Altas"][0] for a in train_dict['val']]
            self.sample_list+=z  
        elif self.split == 'test':
            self.sample_list=[b for a in train_dict['test'] for b in a['OtherImg'] ]
            z=[a["Altas"][0] for a in train_dict['test']]
            self.sample_list+=z

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]#取出名字
        name = case.split(".")[0]
        imgA_path=os.path.join(self.img_path, name+'.npy')
        img = np.load(imgA_path)
        # 简单归一化
        img = img / 255.
        img = img.astype(np.float32)

        # 扩展为3通道的
        img = np.tile(img[:, :, None], (1, 1, 3))
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img)
        parent_dir = os.path.dirname(self.img_path)
        imgB_path = os.path.join(parent_dir,'seg_npy',name+'.npy')
        label = np.load(imgB_path)
        label = np.where(label<=0,0,label)
        label = np.where(label>0,1,label)
        if self.split=="unlabel_train":
            sample = {'image': img}
        else:
            sample = {'image': img, "label": label,"label_x4":cv2.resize(label,(80,40))}

        sample["idx"] = case
        return sample
    




if __name__=='__main__':
    dir_path="zhibiao/zhibiao_11_17_newdata_pseudo_lossis_ce_itr_1000_nocontourloss_up&down_aux3_ce_smoothl1_only_down_VerticalFlip"
    db_train = BaseDataSets(data_file='data_new_2/train_dict1.pkl', img_path="data_new_2/image_crop", zhibiao_dir=dir_path,
                            split="train",baifenbi=0.2,)
    db_unlabel = BaseDataSets(data_file='data_new_2/train_dict1.pkl', img_path="data_new_2/image_crop", zhibiao_dir=dir_path,
                            split="unlabel_train",baifenbi=0.2,)
    unlabel_dataloader = DataLoader(db_unlabel, batch_size=2, shuffle=False,
                           num_workers=1)
    
    print(len(db_train),len(db_unlabel))
    it=itertools.cycle(unlabel_dataloader)

    while True:
        # try:
        x=next(it)
        # except StopIteration:
        #     it=iter(unlabel_dataloader)
        print(x['idx'])
# Expore MGD1k Dataset/Original Images/DMI_OS_UPPER_REFLECTED_IR_16915068.png
