import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
data_img_path = '/mnt/hys/TransMorph_Transformer_for_Medical_Image_Registration-main/MGD/testoutputs/重复测量（hys整理版）/V1/TransMorph_ssim_1_mi_1_diffusion_1_batchsize_8'
image_file_path = os.path.join(data_img_path,'w_m_to_f')
label_file_path = os.path.join(data_img_path,'w_gland_label_m_to_f')
save_path = '/mnt/hys/TransMorph_Transformer_for_Medical_Image_Registration-main/MGD/变形标签叠加图/'


def seg_gland(gland_image,mask_image):
    # 将腺体分割mask应用于睑板腺图
    result_image = cv2.bitwise_and(gland_image, mask_image)
    cv2.imwrite(os.path.join(save_path, f'gland_image.png'), result_image)


# 单纯的将腺体/包含睑板腺区域以及腺体画到原图上并保存
def draw_gland_over_image(save_path,gland_image,mask_image,overlay_color,alpha):
    if (len(overlay_color) == 2):

        overlay_color1  = overlay_color[0]
        overlay_color2 = overlay_color[1]
    
        mask_image1 = np.where(mask_image <= 84, 0, mask_image)
        mask_image1 = np.where(mask_image1> 166, 0, mask_image1)
        mask_image1 = np.where((mask_image1 > 84 )&(mask_image1 <= 166), 255, mask_image1)

        mask_image2 = np.where(mask_image <= 166, 0, mask_image)
        mask_image2 = np.where(mask_image2> 166, 255, mask_image2)

        # 将灰度睑板腺图转换为三通道的彩色图像
        gland_image_color = cv2.cvtColor(gland_image, cv2.COLOR_GRAY2BGR)

        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        overlay_image1 = np.full_like(gland_image_color, overlay_color1, dtype=np.uint8)
        overlay_image2 = np.full_like(gland_image_color, overlay_color2, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        overlay1_channels = cv2.split(overlay_image1)
        overlay2_channels = cv2.split(overlay_image2)
        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        overlayed1_channels = []

        for channel in overlay1_channels:
            overlayed_channel = cv2.bitwise_and(channel, mask_image1)
            overlayed1_channels.append(overlayed_channel)
        # 合并通道
        overlayed1_image = cv2.merge(overlayed1_channels)
        overlayed2_channels = []
        for channel in overlay2_channels:
            overlayed_channel = cv2.bitwise_and(channel, mask_image2)
            overlayed2_channels.append(overlayed_channel)
        # 合并通道
        overlayed2_image = cv2.merge(overlayed2_channels)
        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        result_image = cv2.addWeighted(gland_image_color, 1, overlayed1_image, alpha, 0)
        result_image = cv2.addWeighted(result_image, 1, overlayed2_image, alpha, 0)

        #因为 OpenCV 默认使用 BGR 顺序保存图像，而 Matplotlib 默认使用 RGB 顺序显示图像。所以要进行修改
        # 将 overlayed_image 从 RGB 转换为 BGR

        overlayed_image =  cv2.addWeighted(overlayed1_image, 1, overlayed2_image, 1, 0)
        overlayed_image_bgr = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_path, f'overlayed_image.png'), overlayed_image_bgr)

        # 将 result_image 从 RGB 转换为 BGR
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, f'gland_over_image.png'), result_image_bgr)
    elif (len(overlay_color) == 1):
                # 将灰度睑板腺图转换为三通道的彩色图像
        gland_image_color = cv2.cvtColor(gland_image, cv2.COLOR_GRAY2BGR)
        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        overlay_image = np.full_like(gland_image_color, overlay_color, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        overlay_channels = cv2.split(overlay_image)

        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        overlayed_channels = []
        for channel in overlay_channels:
            overlayed_channel = cv2.bitwise_and(channel, mask_image)
            overlayed_channels.append(overlayed_channel)
        # 合并通道
        overlayed_image = cv2.merge(overlayed_channels)
    
        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        result_image = cv2.addWeighted(gland_image_color, 1, overlayed_image, alpha, 0)

        #因为 OpenCV 默认使用 BGR 顺序保存图像，而 Matplotlib 默认使用 RGB 顺序显示图像。所以要进行修改
        # 将 overlayed_image 从 RGB 转换为 BGR

        overlayed_image_bgr = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_path, f'overlayed_image.png'), overlayed_image_bgr)

        # 将 result_image 从 RGB 转换为 BGR
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, f'gland_over_image.png'), result_image_bgr)



# 读取两个文件夹，将文件夹中的对应的图像和标签画出来
def draw_seg_to_img_cv2_list(image_file_path,label_file_path,save_path, overlay_color = (255,100,100),alpha=0.3):
    if not os.path.exists(image_file_path):
        print("图像文件夹不存在")
        return 0
    else:
        # 读取路径 a 下的所有 jpg 图片
        image_list = [file for file in os.listdir(image_file_path) if file.endswith('.jpg')]
        for img_name in image_list :
            
            # 读取睑板腺图和腺体分割mask
            
            if not os.path.exists(os.path.join(label_file_path,img_name)):
                print(img_name+"对应标签不存在")
                continue
            image = cv2.imread(os.path.join(image_file_path,img_name), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(label_file_path,img_name), cv2.IMREAD_GRAYSCALE)
            draw_gland_over_image(save_path,image,label,overlay_color,alpha)


# 可以实现将两张图像的各自的标签叠加到图像上,并最终叠加到一张图中,保存并返回.具体就是变形后的标签画到变形后的图像上,目标图像的标签画到目标图像上,然后再将变形图像和目标图像叠加,观察其配准的效果
def draw_gland_over_image_and_fix_moving_fixed_imgs(save_path,moving_gland_image,moving_mask_image,fixed_gland_image,fixed_mask_image,moving_name,fixed_name,overlay_color,alpha):
    if (len(overlay_color) == 4):

        overlay_color1  = overlay_color[0]
        overlay_color2 = overlay_color[1]

        overlay_color3  = overlay_color[2]
        overlay_color4 = overlay_color[3]
    
        moving_mask_image1 = np.where(moving_mask_image <= 84, 0, moving_mask_image)
        moving_mask_image1 = np.where(moving_mask_image1> 166, 0, moving_mask_image1)
        moving_mask_image1 = np.where((moving_mask_image1 > 84 )&(moving_mask_image1 <= 166), 255, moving_mask_image1)

    
        moving_mask_image2 = np.where(moving_mask_image <= 166, 0, moving_mask_image)
        moving_mask_image2 = np.where(moving_mask_image2> 166, 255, moving_mask_image2)

        # 将灰度睑板腺图转换为三通道的彩色图像
        moving_gland_image_color = cv2.cvtColor(moving_gland_image, cv2.COLOR_GRAY2BGR)



        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        moving_overlay_image1 = np.full_like(moving_gland_image_color, overlay_color1, dtype=np.uint8)
        moving_overlay_image2 = np.full_like(moving_gland_image_color, overlay_color2, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        moving_overlay1_channels = cv2.split(moving_overlay_image1)
        moving_overlay2_channels = cv2.split(moving_overlay_image2)
        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        moving_overlayed1_channels = []

        for channel in moving_overlay1_channels:
            overlayed_channel = cv2.bitwise_and(channel, moving_mask_image1)
            moving_overlayed1_channels.append(overlayed_channel)
        # 合并通道
        moving_overlayed1_image = cv2.merge(moving_overlayed1_channels)
        moving_overlayed2_channels = []
        for channel in moving_overlay2_channels:
            overlayed_channel = cv2.bitwise_and(channel, moving_mask_image2)
            moving_overlayed2_channels.append(overlayed_channel)
        # 合并通道
        moving_overlayed2_image = cv2.merge(moving_overlayed2_channels)
        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        moving_result_image = cv2.addWeighted(moving_gland_image_color, 1, moving_overlayed1_image, alpha, 0)
        moving_result_image = cv2.addWeighted(moving_result_image, 1, moving_overlayed2_image, alpha, 0)
    


        '''
        fix img
        '''

        fixed_mask_image1 = np.where(fixed_mask_image <= 84, 0, fixed_mask_image)
        fixed_mask_image1 = np.where(fixed_mask_image1> 166, 0, fixed_mask_image1)
        fixed_mask_image1 = np.where((fixed_mask_image1 > 84 )&(fixed_mask_image1 <= 166), 255, fixed_mask_image1)

    
        fixed_mask_image2 = np.where(fixed_mask_image <= 166, 0, fixed_mask_image)
        fixed_mask_image2 = np.where(fixed_mask_image2> 166, 255, fixed_mask_image2)

        # 将灰度睑板腺图转换为三通道的彩色图像
        fixed_gland_image_color = cv2.cvtColor(fixed_gland_image, cv2.COLOR_GRAY2BGR)



        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        fixed_overlay_image1 = np.full_like(fixed_gland_image_color, overlay_color3, dtype=np.uint8)
        fixed_overlay_image2 = np.full_like(fixed_gland_image_color, overlay_color4, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        fixed_overlay1_channels = cv2.split(fixed_overlay_image1)
        fixed_overlay2_channels = cv2.split(fixed_overlay_image2)
        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        fixed_overlayed1_channels = []

        for channel in fixed_overlay1_channels:
            overlayed_channel = cv2.bitwise_and(channel, fixed_mask_image1)
            fixed_overlayed1_channels.append(overlayed_channel)
        # 合并通道
        fixed_overlayed1_image = cv2.merge(fixed_overlayed1_channels)
        fixed_overlayed2_channels = []
        for channel in fixed_overlay2_channels:
            overlayed_channel = cv2.bitwise_and(channel, fixed_mask_image2)
            fixed_overlayed2_channels.append(overlayed_channel)
        # 合并通道
        fixed_overlayed2_image = cv2.merge(fixed_overlayed2_channels)
        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        fixed_result_image = cv2.addWeighted(fixed_gland_image_color, 1, fixed_overlayed1_image, alpha, 0)
        fixed_result_image = cv2.addWeighted(fixed_result_image, 1, fixed_overlayed2_image, alpha, 0)

        '''
            变形图像和移动图像叠加起来
        '''
        result = cv2.addWeighted(moving_result_image, 0.5, fixed_result_image, 0.5, 0)


        # 将 result_image 从 RGB 转换为 BGR
        result_image_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if not os.path.exists(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland')):
            os.makedirs(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland'))
        cv2.imwrite(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland', moving_name[:-4] + '_' + fixed_name[:-4] + '.jpg'), result_image_bgr)
        return result
    elif  (len(overlay_color) == 2):
        
        overlay_color1  = overlay_color[0]
        overlay_color2 = overlay_color[1]

    
        moving_mask_image = np.where(moving_mask_image <= 128, 0, moving_mask_image)
        moving_mask_image = np.where(moving_mask_image> 128, 255, moving_mask_image)

        # 将灰度睑板腺图转换为三通道的彩色图像
        moving_gland_image_color = cv2.cvtColor(moving_gland_image, cv2.COLOR_GRAY2BGR)



        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        moving_overlay_image = np.full_like(moving_gland_image_color, overlay_color1, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        moving_overlay_channels = cv2.split(moving_overlay_image)
        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        moving_overlayed_channels = []

        for channel in moving_overlay_channels:
            overlayed_channel = cv2.bitwise_and(channel, moving_mask_image)
            moving_overlayed_channels.append(overlayed_channel)
        # 合并通道
        moving_overlayed_image = cv2.merge(moving_overlayed_channels)

        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        moving_result_image = cv2.addWeighted(moving_gland_image_color, 1, moving_overlayed_image, alpha, 0)
    
        '''
        fix img
        '''

    
        fixed_mask_image = np.where(fixed_mask_image <= 128, 0, fixed_mask_image)
        fixed_mask_image = np.where(fixed_mask_image> 128, 255, fixed_mask_image)

        # 将灰度睑板腺图转换为三通道的彩色图像
        fixed_gland_image_color = cv2.cvtColor(fixed_gland_image, cv2.COLOR_GRAY2BGR)



        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        fixed_overlay_image = np.full_like(fixed_gland_image_color, overlay_color2, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        fixed_overlay_channels = cv2.split(fixed_overlay_image)
        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        fixed_overlayed_channels = []

        for channel in fixed_overlay_channels:
            overlayed_channel = cv2.bitwise_and(channel, fixed_mask_image)
            fixed_overlayed_channels.append(overlayed_channel)
        # 合并通道
        fixed_overlayed_image = cv2.merge(fixed_overlayed_channels)
        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        fixed_result_image = cv2.addWeighted(fixed_gland_image_color, 1, fixed_overlayed_image, alpha, 0)

        '''
            变形图像和移动图像叠加起来
        '''
        result = cv2.addWeighted(moving_result_image, 0.5, fixed_result_image, 0.5, 0)


        # 将 result_image 从 RGB 转换为 BGR
        result_image_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if not os.path.exists(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland')):
            os.makedirs(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland'))
        cv2.imwrite(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland', moving_name[:-4] + '_' + fixed_name[:-4] + '.jpg'), result_image_bgr)
        # ----------------计算腺体的指标----------------
        # 目标图像的标签
        tmp_fixed_gland = np.where(fixed_mask_image==255,1,fixed_mask_image)
        # 变形得到的标签
        tmpgland = np.where(moving_mask_image==255,1,moving_mask_image)
        # 判断分割图与标签腺体比较的布尔值
        comparison = tmpgland == tmp_fixed_gland
        tp = np.sum(np.logical_and(comparison, tmpgland == 1)) #变形标签为腺体且此位置目标图像的标签也为腺体的数量。（）
        # 计算fn，标签为腺体，但错分为背景的数目
        fn = np.sum(tmp_fixed_gland == 1) - tp #实际上是腺体，但却背景的数量
        fp = np.sum(tmpgland == 1) - tp #实际上是背景，但是腺体
        iou = tp/(fn+tp+fp)
        recall = tp/(tp+fn)
        dice = 2*tp / (2*tp + fn + fp)

        # ----------------计算腺体的指标----------------

        return result,iou,recall,dice

def draw_gland_over_image_and_fix_moving_fixed_imgs_without_mask(save_path,moving_gland_image,moving_mask_image,fixed_gland_image,fixed_mask_image,moving_name,fixed_name,overlay_color,alpha):
        overlay_color1  = overlay_color[0]
        overlay_color2 = overlay_color[1]

    
        moving_mask_image = np.where(moving_mask_image <= 0, 0, moving_mask_image)
        moving_mask_image = np.where(moving_mask_image> 0, 255, moving_mask_image)

        # 将灰度睑板腺图转换为三通道的彩色图像
        moving_gland_image_color = cv2.cvtColor(moving_gland_image, cv2.COLOR_GRAY2BGR)



        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        moving_overlay_image = np.full_like(moving_gland_image_color, overlay_color1, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        moving_overlay_channels = cv2.split(moving_overlay_image)
        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        moving_overlayed_channels = []

        for channel in moving_overlay_channels:
            overlayed_channel = cv2.bitwise_and(channel, moving_mask_image)
            moving_overlayed_channels.append(overlayed_channel)
        # 合并通道
        moving_overlayed_image = cv2.merge(moving_overlayed_channels)

        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        moving_result_image = cv2.addWeighted(moving_gland_image_color, 1, moving_overlayed_image, alpha, 0)
    
        '''
        fix img
        '''

    
        fixed_mask_image = np.where(fixed_mask_image <= 0, 0, fixed_mask_image)
        fixed_mask_image = np.where(fixed_mask_image> 0, 255, fixed_mask_image)

        # 将灰度睑板腺图转换为三通道的彩色图像
        fixed_gland_image_color = cv2.cvtColor(fixed_gland_image, cv2.COLOR_GRAY2BGR)



        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        fixed_overlay_image = np.full_like(fixed_gland_image_color, overlay_color2, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        fixed_overlay_channels = cv2.split(fixed_overlay_image)
        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        fixed_overlayed_channels = []

        for channel in fixed_overlay_channels:
            overlayed_channel = cv2.bitwise_and(channel, fixed_mask_image)
            fixed_overlayed_channels.append(overlayed_channel)
        # 合并通道
        fixed_overlayed_image = cv2.merge(fixed_overlayed_channels)
        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        fixed_result_image = cv2.addWeighted(fixed_gland_image_color, 1, fixed_overlayed_image, alpha, 0)

        '''
            变形图像和移动图像叠加起来
        '''
        result = cv2.addWeighted(moving_result_image, 0.5, fixed_result_image, 0.5, 0)


        # 将 result_image 从 RGB 转换为 BGR
        result_image_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if not os.path.exists(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland')):
            os.makedirs(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland'))
        cv2.imwrite(os.path.join(save_path, 'overlay_moving_and_fixed_img_with_gland', moving_name + '_' + fixed_name + '.jpg'), result_image_bgr)
        return  result_image_bgr


# 将生成的伪标签画到图像上（pseudo_gland_image腺体，pseudo_bg_imgage背景）
def draw_pseudo_gland_over_image(save_path,gland_image,pseudo_gland_image,pseudo_bg_imgage,overlay_color,alpha,add_name):
    if (len(overlay_color) == 2):

        overlay_color1  = overlay_color[0]
        overlay_color2 = overlay_color[1]
    
        mask_image1 = pseudo_gland_image
        mask_image2 = pseudo_bg_imgage


        # 将灰度睑板腺图转换为三通道的彩色图像
        gland_image_color = cv2.cvtColor(gland_image, cv2.COLOR_GRAY2BGR)

        # 创建一个与灰度睑板腺图像大小相同的彩色图像，填充为浅色
        overlay_image1 = np.full_like(gland_image_color, overlay_color1, dtype=np.uint8)
        overlay_image2 = np.full_like(gland_image_color, overlay_color2, dtype=np.uint8)

        # 将 overlay_image 拆分为三个通道
        overlay1_channels = cv2.split(overlay_image1)
        overlay2_channels = cv2.split(overlay_image2)
        # 如果把除去睑板腺的睑板区域也作为一类的话。

        # 将每个通道与腺体分割 mask 进行按位与操作
        overlayed1_channels = []

        #画腺体伪标签
        for channel in overlay1_channels:
            overlayed_channel = cv2.bitwise_and(channel, mask_image1)
            overlayed1_channels.append(overlayed_channel)
        # 合并通道
        overlayed1_image = cv2.merge(overlayed1_channels)
        
        # 画背景伪标签
        overlayed2_channels = []
        for channel in overlay2_channels:
            overlayed_channel = cv2.bitwise_and(channel, mask_image2)
            overlayed2_channels.append(overlayed_channel)
        # 合并通道
        overlayed2_image = cv2.merge(overlayed2_channels)
        # 使用图像融合函数将浅色叠加图像与灰度睑板腺图像进行融合
        result_image = cv2.addWeighted(gland_image_color, 1, overlayed1_image, alpha, 0)
        result_image = cv2.addWeighted(result_image, 1, overlayed2_image, alpha, 0)

        #因为 OpenCV 默认使用 BGR 顺序保存图像，而 Matplotlib 默认使用 RGB 顺序显示图像。所以要进行修改
        # 将 overlayed_image 从 RGB 转换为 BGR

        overlayed_image =  cv2.addWeighted(overlayed1_image, 1, overlayed2_image, 1, 0)
        overlayed_image_bgr = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)
                    
        cv2.imwrite(os.path.join(save_path, f'{add_name}.png'), overlayed_image_bgr)

        # # 将 result_image 从 RGB 转换为 BGR
        # result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(save_path, f'{add_name}.png'), result_image_bgr)
        
        



gland_image = '/mnt/hys/Datasets/重复测量（hys整理版）/img_crop_npy/男_柯元礼_右_2024_01_26-2睑板腺概览.npy'
mask_image = '/mnt/hys/Datasets/重复测量（hys整理版）/seg_npy/男_柯元礼_右_2024_01_26-2睑板腺概览.npy'
gland2_image = '/mnt/hys/Datasets/重复测量（hys整理版）/img_crop_npy/男_柯元礼_右_2024_01_26-3睑板腺概览.npy'
mask2_image = '/mnt/hys/Datasets/重复测量（hys整理版）/seg_npy/男_柯元礼_右_2024_01_26-3睑板腺概览.npy'
save_path ='/home/hys/code/ztest'
overlay_color = [(128, 0, 128),(0,255,255)] #紫色，青色
alpha = 0.5
image = np.load(gland_image)
label =  np.load(mask_image)
image2 = np.load(gland2_image)
label2 =  np.load(mask2_image)

draw_gland_over_image_and_fix_moving_fixed_imgs_without_mask(save_path,image,label,image2,label2,'2','1',overlay_color,alpha)