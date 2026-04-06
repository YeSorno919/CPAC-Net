import torch
import torch.nn as nn
import torch.nn.functional as nnf


# 和Fourier-Net 2d中的SpatialTransform不太一样. 需要修改?
class SpatialTransformer_2d(nn.Module):
    def __init__(self,islabel):
        super(SpatialTransformer_2d, self).__init__()
        self.islabel = islabel
    def forward(self, src, flow,mode = 'bilinear'):
        if(self.islabel):
            mode = 'nearest'

        shape = flow.shape[2:]
        # print("shape",shape) #shape torch.Size([350, 740])
        vectors = [torch.arange(0, s) for s in shape] # vectors由两个列表构成，其中分别包含了从0到高度和宽度的序列
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # grid形状为（2，高度，宽度）
        grid = torch.unsqueeze(grid, 0)  # add batch 添加batch这个维度 得到（1，2，高度，宽度）
        grid = grid.type(torch.FloatTensor)
        device = torch.device('cuda:1')
        grid = grid.to(device)
        #会将每个网格点按照flow的偏移量进行调整
        new_locs = grid + flow
        # 每个维度,新位置的值归一化到-1,1之间
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)
        # 根据输入特征图的维度數量,对新位置进行维度排列和调整
        # 二维图像,维度调整为(batch,height,width,channels) 原本是(b,c,h,w).并将最后一个维度的顺序调整为[1,0]
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        #3维特征图,调整为(batch,depth,height,width,channels),并将最后一个维度的顺序调整为[2,1,0]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]
        #使用pytorch的grid_sample根据新位置对输入特征图进行采样,得到经过空间变换后的输出结果.
        output = nnf.grid_sample(src, new_locs, mode=mode)
        if(self.islabel):
            output[::,0]=1-output[::,1]
        return output

    
    
class Re_SpatialTransformer_2d(nn.Module):
    def __init__(self,islabel):
        super(Re_SpatialTransformer_2d, self).__init__()
        self.islabel =islabel

        self.stn = SpatialTransformer_2d(self.islabel)

    def forward(self, src, flow, mode='bilinear'):
        # 需要修改?  为何是self.stn(flow, flow, mode='bilinear'),两个flow
        flow = -1 * self.stn(flow, flow, mode='bilinear')

        return self.stn(src, flow, mode)