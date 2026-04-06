import os
import numpy as np
from torch import nn
import torch

import torch.nn.functional as F
# SE模块，能够明显的模拟通道之间的相互以来性，自适应的重新校准通道特征相应。
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
# 位置注意力机制
class PAM(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(PAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // ratio
        
        # 可学习的参数 gamma
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 卷积层
        self.conv_b = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_c = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        batch_size, height, width = x.size()[0], x.size()[2], x.size()[3]

        # 卷积层
        b = self.conv_b(x)
        c = self.conv_c(x)
        d = self.conv_d(x)

        # Reshape and transpose
        vec_b = b.view(batch_size, self.inter_channels, -1) # b*(c/r)*(h*w)
        vec_c = c.view(batch_size, self.inter_channels, -1).permute(0, 2, 1) # b*(h*w)*(c/r)
        vec_d = d.view(batch_size, self.in_channels, -1).permute(0, 2, 1) #b*(h*w)*(c)

        # 计算注意力图
        bcT = torch.bmm(vec_c, vec_b) #b*(h*w)*(h*w)
        attention = F.softmax(bcT, dim=-1) #b*(h*w)*(h*w) 对每个位置计算其与其他位置之间的相对权重
        
        # 加权求和
        bcTd = torch.bmm(attention, vec_d).permute(0, 2, 1)
        bcTd = bcTd.view(batch_size, self.in_channels, height, width)

        # 输出
        out = self.gamma * bcTd + x
        return out


class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 

# https://openaccess.thecvf.com/content/ACCV2024/papers/Wang_PlainUSR_Chasing_Faster_ConvNet_for_Efficient_Super-Resolution_ACCV_2024_paper.pdf
# ACCV 2024 局部重要性注意力机制    
class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            # 调整通道数
            nn.Conv2d(channels, f, 1),
            # 软池化捕捉重要性信息
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            # 通过sigmoid激活函数来生成权重
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        # 对输入的第一通道应用门控机制
        g = self.gate(x[:,:1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * w * g #(w + g) #self.gate(x, w) 

class LocalAttentionSpeed(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        f = f
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, channels, 3, 1, 1),
            # SoftPooling2D(7, stride=3),
            # nn.Conv2d(f, channels, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(f, 1, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:,:1])
        w = self.body(x) 

        return x * g * w #(w + g) #self.gate(x, w)    



    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A = np.random.rand(1, 16, 176, 376)
    # # A = np.ones(shape=(3, 2, 2, 3), dtype=np.float32)
    # # print(A)
    # A = A.astype(dtype=np.float32)
    # A = torch.from_numpy(A)
    # # print(A.shape)
    # # conv0 = DSConv(
    # #     in_ch=5,
    # #     out_ch=10,
    # #     kernel_size=15,
    # #     extend_scope=1,
    # #     morph=0,
    # #     if_offset=True,
    # #     device=device)

    # # out = conv0(A)
    # # print(out.shape)
    # # print(out)
    # # net = SELayer(channel=16, reduction=16)
    # # if torch.cuda.is_available():
    # #     A = A.to(device)
    # #     net = net.to(device)
    # # out = net(A)
    # # print(out.shape)
    # # print(out)
    
    # net = PAM(in_channels=16, ratio=8)
    # if torch.cuda.is_available():
    #     A = A.to(device)
    #     net = net.to(device)
    # out = net(A)
    # print(out.shape)
    # print(out)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.rand(1, 1, 350, 740)
    # A = np.ones(shape=(3, 2, 2, 3), dtype=np.float32)
    # print(A)
    LA = LocalAttention(1)


    

    if torch.cuda.is_available():
        A = A.to(device)
        LA = LA.to(device)
    out = LA(A)
    print(out.shape)
    print(out)