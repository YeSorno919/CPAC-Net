import random
import numpy as np
import torch
# from scipy.io import loadmat
from torch import nn
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

import torch.nn.functional as nnf
class MirrorTransform(object):
    def augment_mirroring(self, data, code=(1, 1, 1)):
        if code[0] == 1:
            data = self.flip(data, 2)
        if code[1] == 1:
            data = self.flip(data, 3)
        if code[2] == 1:
            data = self.flip(data, 4)
        return data

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def rand_code(self):
        code = []
        for i in range(3):
            if np.random.uniform() < 0.5:
                code.append(1)
            else:
                code.append(0)
        return code

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear', padding_mode='zeros'):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)

        device = torch.device('cuda:1')
        grid = grid.to(device)

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return nnf.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode)
# 需要修改
class SpatialTransform_2d(object):
    def __init__(self, do_rotation=True, angle = (0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25)):
        self.do_rotation = do_rotation
        self.angle = angle
        self.do_scale = do_scale
        self.scale = scale
        self.stn = SpatialTransformer() #这里的变形是用本文件中的SpatialTransformer,注意STN.py也有一个SpatialTransformer,两者其实一样

    def augment_spatial(self, data, code, mode='bilinear'):
        data = self.stn(data, code, mode=mode, padding_mode='zeros')
        return data

    # 生成随机坐标变换
    def rand_coords(self, patch_size):
        coords = self.create_zero_centered_coordinate_mesh(patch_size)
        if self.do_rotation:
            # 进行2维旋转变换
            angle = np.random.uniform(self.angle[0], self.angle[1])
            coords = self.rotate_coords_2d(coords, angle)
        # 如果尺度变换,则在范围内随机旋转尺度因子sc,进行尺度变换
        if self.do_scale:
            sc = np.random.uniform(self.scale[0], self.scale[1])
            coords = self.scale_coords(coords, sc)
        # 中心坐标ctr,值为patch_size的一半
        ctr = np.asarray([patch_size[0]//2, patch_size[1]//2])
        # 生成包含所有网格点的坐标网络grid,大小同patch_size
        grid = np.where(np.ones(patch_size)==1)
        grid = np.concatenate([grid[0].reshape((1,)+patch_size), grid[1].reshape((1,)+patch_size)], axis=0)
        grid = grid.astype(np.float32)
        # 坐标平移
        coords += ctr[:, np.newaxis, np.newaxis] - grid
        coords = coords.astype(np.float32)
        # 转变成 torch.Tensor 格式
        coords = torch.from_numpy(coords[np.newaxis, :, :, :])
        if torch.cuda.is_available():
            coords = coords.cuda()
        return coords

    # 创建一个以原点为中心的坐标网络，坐标值是相对于中心点的偏移量。
    def create_zero_centered_coordinate_mesh(self, shape):
        # 根据shape参数,生成包含每个维度索引值的元组tmp
        tmp = tuple([np.arange(i) for i in shape])
        # np.meshgrid 函数根据 tmp 中的索引值生成一个多维坐标网格 coords。参数 indexing='ij' 表示使用以索引值为基准的多维网格。
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
        # 对于coords的每个维度d，减去对应维度上的中心坐标实现将坐标网格以原点为中心的目的。
        for d in range(len(shape)):
            coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
        return coords

    # 二维旋转变换。
    def rotate_coords_2d(self,coords, angle):
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
        return coords



    # 水平，垂直尺度变换,返回尺度变换后的坐标
    def scale_coords(self, coords, scale):
        if isinstance(scale, (tuple, list, np.ndarray)):
            assert len(scale) == len(coords)
            for i in range(len(scale)):
                coords[i] *= scale[i]
        else:
            coords *= scale
        return coords


import torch
import torchvision.transforms.functional as F

# 增强方式有随机的概率进行
class SpatialTransform_2d_random(object):
    def __init__(self, do_rotation=True, angle = (0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25),do_flip = True,random_percent=0.5):
        self.do_rotation = do_rotation
        self.angle = angle
        self.do_scale = do_scale
        self.scale = scale
        self.random_percent=random_percent
        self.do_flip=do_flip
        self.stn = SpatialTransformer() #这里的变形是用本文件中的SpatialTransformer,注意STN.py也有一个SpatialTransformer,两者其实一样

    def augment_spatial(self, data, code, mode='bilinear'):
        data = self.stn(data, code, mode=mode, padding_mode='zeros')
        return data

    # 生成随机坐标变换
    def rand_coords(self, patch_size):
        coords = self.create_zero_centered_coordinate_mesh(patch_size)
        if self.do_rotation:
            # 进行2维旋转变换
            if(np.random.rand() > self.random_percent):
                angle = np.random.uniform(self.angle[0], self.angle[1])
                coords = self.rotate_coords_2d(coords, angle)
        # 如果尺度变换,则在范围内随机旋转尺度因子sc,进行尺度变换
        if self.do_scale:
            if(np.random.rand() > self.random_percent):
                sc = np.random.uniform(self.scale[0], self.scale[1])
                coords = self.scale_coords(coords, sc)

        # 添加翻转功能
        if self.do_flip:
            # 随机决定是否进行水平翻转
            if np.random.rand() > self.random_percent:
                coords[1] = -coords[1]  # 水平翻转
            # 随机决定是否进行垂直翻转
            if np.random.rand() > self.random_percent:
                coords[0] = -coords[0]  # 垂直翻转
                
        # 中心坐标ctr,值为patch_size的一半
        ctr = np.asarray([patch_size[0]//2, patch_size[1]//2])
        # 生成包含所有网格点的坐标网络grid,大小同patch_size
        grid = np.where(np.ones(patch_size)==1)
        grid = np.concatenate([grid[0].reshape((1,)+patch_size), grid[1].reshape((1,)+patch_size)], axis=0)
        grid = grid.astype(np.float32)
        # 坐标平移
        coords += ctr[:, np.newaxis, np.newaxis] - grid
        coords = coords.astype(np.float32)
        # 转变成 torch.Tensor 格式
        coords = torch.from_numpy(coords[np.newaxis, :, :, :])
        return coords

    # 创建一个以原点为中心的坐标网络，坐标值是相对于中心点的偏移量。
    def create_zero_centered_coordinate_mesh(self, shape):
        # 根据shape参数,生成包含每个维度索引值的元组tmp
        tmp = tuple([np.arange(i) for i in shape])
        # np.meshgrid 函数根据 tmp 中的索引值生成一个多维坐标网格 coords。参数 indexing='ij' 表示使用以索引值为基准的多维网格。
        coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
        # 对于coords的每个维度d，减去对应维度上的中心坐标实现将坐标网格以原点为中心的目的。
        for d in range(len(shape)):
            coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
        return coords

    # 二维旋转变换。
    def rotate_coords_2d(self,coords, angle):
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
        return coords



    # 水平，垂直尺度变换,返回尺度变换后的坐标
    def scale_coords(self, coords, scale):
        if isinstance(scale, (tuple, list, np.ndarray)):
            assert len(scale) == len(coords)
            for i in range(len(scale)):
                coords[i] *= scale[i]
        else:
            coords *= scale
        return coords
    
# def img_aug_identity(img, scale=None):
#     return img


# def img_aug_autocontrast(img, scale=None):
#     return ImageOps.autocontrast(img)


# def img_aug_equalize(img, scale=None):
#     return ImageOps.equalize(img)


# def img_aug_invert(img, scale=None):
#     return ImageOps.invert(img)


# def img_aug_blur(img, scale=[0.1, 2.0]):
#     assert scale[0] < scale[1]
#     sigma = np.random.uniform(scale[0], scale[1])
#     # print(f"sigma:{sigma}")
#     return img.filter(ImageFilter.GaussianBlur(radius=sigma))


# def img_aug_contrast(img, scale=[0.05, 0.95]):
#     min_v, max_v = min(scale), max(scale)
#     v = float(max_v - min_v)*random.random()
#     v = max_v - v
#     # # print(f"final:{v}")
#     # v = np.random.uniform(scale[0], scale[1])
#     return ImageEnhance.Contrast(img).enhance(v)


# def img_aug_brightness(img, scale=[0.05, 0.95]):
#     min_v, max_v = min(scale), max(scale)
#     v = float(max_v - min_v)*random.random()
#     v = max_v - v
#     # print(f"final:{v}")
#     return ImageEnhance.Brightness(img).enhance(v)


# def img_aug_color(img, scale=[0.05, 0.95]):
#     min_v, max_v = min(scale), max(scale)
#     v = float(max_v - min_v)*random.random()
#     v = max_v - v
#     # print(f"final:{v}")
#     return ImageEnhance.Color(img).enhance(v)


# def img_aug_sharpness(img, scale=[0.05, 0.95]):
#     min_v, max_v = min(scale), max(scale)
#     v = float(max_v - min_v)*random.random()
#     v = max_v - v
#     # print(f"final:{v}")
#     return ImageEnhance.Sharpness(img).enhance(v)


# def img_aug_hue(img, scale=[0, 0.5]):
#     min_v, max_v = min(scale), max(scale)
#     v = float(max_v - min_v)*random.random()
#     v += min_v
#     if np.random.random() < 0.5:
#         hue_factor = -v
#     else:
#         hue_factor = v
#     # print(f"Final-V:{hue_factor}")
#     input_mode = img.mode
#     if input_mode in {"L", "1", "I", "F"}:
#         return img
#     h, s, v = img.convert("HSV").split()
#     np_h = np.array(h, dtype=np.uint8)
#     # uint8 addition take cares of rotation across boundaries
#     with np.errstate(over="ignore"):
#         np_h += np.uint8(hue_factor * 255)
#     h = Image.fromarray(np_h, "L")
#     img = Image.merge("HSV", (h, s, v)).convert(input_mode)
#     return img


# def img_aug_posterize(img, scale=[4, 8]):
#     min_v, max_v = min(scale), max(scale)
#     v = float(max_v - min_v)*random.random()
#     # print(min_v, max_v, v)
#     v = int(np.ceil(v))
#     v = max(1, v)
#     v = max_v - v
#     # print(f"final:{v}")
#     return ImageOps.posterize(img, v)


# def img_aug_solarize(img, scale=[1, 256]):
#     min_v, max_v = min(scale), max(scale)
#     v = float(max_v - min_v)*random.random()
#     # print(min_v, max_v, v)
#     v = int(np.ceil(v))
#     v = max(1, v)
#     v = max_v - v
#     # print(f"final:{v}")
#     return ImageOps.solarize(img, v)    
# def img_aug_identity(img, scale=None):
#     return img

# 原始图像
def img_aug_identity(img, scale=None):
    return img

# 随机对比度调整(默认自动调整)
def img_aug_autocontrast(img, scale=None):
    # 先将torch张量转换为PIL图像，进行操作后再转换回torch张量
    if scale==None:
        pil_img = F.to_pil_image(img)
        pil_img = F.autocontrast(pil_img)
    else:
        """
        对输入的torch张量图像进行对比度调整
        :param img: 输入的torch张量图像
        :param scale: 对比度调整因子的范围
        :return: 调整对比度后的torch张量图像
        """
        # 从指定范围随机选择一个对比度调整因子
        contrast_factor = random.uniform(scale[0], scale[1])
        # 调整图像的对比度
        pil_img = F.to_pil_image(img)
        pil_img = F.adjust_contrast(pil_img, contrast_factor)
    return F.to_tensor(pil_img)


# 直方图均衡化
def img_aug_equalize(img, scale=None):
    pil_img = F.to_pil_image(img)
    pil_img = F.equalize(pil_img)
    return F.to_tensor(pil_img)

# # 像素翻转
# def img_aug_invert(img, scale=None):
#     return 1 - img

# 高斯模糊
def img_aug_blur(img, scale=[0.5, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    img = F.gaussian_blur(img, kernel_size=3, sigma=sigma)
    return img

# 调整图像对比度
def img_aug_contrast(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    img = F.adjust_contrast(img, v)
    return img

# 调整亮度
def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    img = F.adjust_brightness(img, v)
    return img

# 调整图像的色彩饱和度
def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    img = F.adjust_saturation(img, v)
    return img

# 调整图像的锐度
def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    img = F.adjust_sharpness(img, v)
    return img

# 调整图像的色调
def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    img = F.adjust_hue(img, hue_factor)
    return img

# 对图像进行色调分离操作
def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    img = F.posterize((img * 255).byte(), v).float() / 255
    return img

# 对图像进行曝光反转（Solarize）操作
def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    img = F.solarize((img * 255).byte(), v).float() / 255
    return img

def get_augment_list(flag_using_wide=False):  
    if flag_using_wide:
        l = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.1, 2.0]),
        (img_aug_contrast, [0.1, 1.8]),
        (img_aug_brightness, [0.1, 1.8]),
        (img_aug_color, [0.1, 1.8]),
        (img_aug_sharpness, [0.1, 1.8]),
        (img_aug_posterize, [2, 8]),
        (img_aug_solarize, [1, 256]),
        (img_aug_hue, [0, 0.5])
        ]
    else:
        l = [
            (img_aug_identity, None),# 原始图像
            # (img_aug_autocontrast, [0.5,1.2]),# 随机对比度调整,(img_aug_autocontrast, [0.5,1.5]),# 随机对比度调整
            (img_aug_equalize, None), # 直方图均衡化 图怪
            (img_aug_blur, [0.5, 2.0]), #高斯模糊
            (img_aug_contrast, [1.5, 1.2]),# 调整图像对比度 
            (img_aug_brightness, [0.5, 1.5]),# 调整亮度
            # (img_aug_color, [0.05, 1.5]),# 调整图像的色彩饱和度，没变化
            (img_aug_sharpness, [0.5, 3.0]),# 调整图像的锐度
            #(img_aug_posterize, [4, 8]),#对图像进行色调分离操作，让图像由几种有限的颜色组成
            # (img_aug_solarize, [1, 256]),#对图像进行曝光反转（Solarize）操作。 值如何设置，很怪
            (img_aug_hue, [0, 0.5])# 调整图像的色调
        ]
    return l
