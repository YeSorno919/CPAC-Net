from skimage import measure
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from unet_model import *
from torch.utils.data import DataLoader
from mt_Dataset import *
from Vnet_2d import VNet2d
# from mt_Dataset_Kvasir import *
import medpy
# from networks.vision_transformer import SwinUnet as ViT_seg
from unet_unimatch import UNet_unimatch
import medpy.metric
from attention_unet_model import *
from fvcore.nn import FlopCountAnalysis
from unet_cct import UNet_CCT
from networks.vision_transformer import SwinUnet as ViT_seg
from thop import profile
from mt_Dataset import denormalize


mean = [0.485]
std = [0.229]
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
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)f
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
        # 二分类的acc
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

def cal_zhibiao_gate_onecoder(dir_path,img_path,data_file,model1_path,model2_path,data_type,device='cpu',add_content=''):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''
    model_name=model1_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    encoder=UNet_encoder(3,2,True,True)
    decoder=UNet_decoder_gate(2,True)
    encoder.load_state_dict(torch.load(model1_path,map_location=device))
    decoder.load_state_dict(torch.load(model2_path,map_location=device))
    encoder.to(device).eval()
    decoder.to(device).eval()


    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (740, 350))
            # img_label=cv2.resize(img_label, (740, 350))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            avg_iou=sum(iou_list)/len(iou_list)
            hd95_list.append(medpy.metric.binary.hd95(output2,img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc)+'\n')
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc]




def cal_zhibiao(dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSets(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSets(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSets(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet(3, 2, bilinear=True,aux=False)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        # print(sampled_batch['image'].size())
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())

    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    

    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
def cal_zhibiao_dantongdao(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resize(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resize(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resize(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(1, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet(1, 2, bilinear=True,aux=False)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        # print(sampled_batch['image'].size())
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        
        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            # 将结果画出来
            if(save_img):
                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((denormalize(input_img,mean=mean,std=std).numpy()[j])*255).astype(np.uint8).T[:350,:740]
                # input_img=np.resize(input_img,(350,740))

                lab = (img_label*255).astype(np.uint8).T[:350,:740]
                # lab = np.resize(lab,(350,740))
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = (output2*255).T[:350,:740]
                # pred_img = np.resize(pred_img,(350,740))
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
                presave = Image.fromarray(pred_img.astype(np.uint8))
                presavepath = os.path.join(dir_path,str(fold_num)+'/seg')
                if not os.path.exists(presavepath):
                    os.makedirs(presavepath)
                presave.save(os.path.join(presavepath, sampled_batch["idx"][j][:-4] + '.jpg'))
                
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
def cal_zhibiao_CNN_Transformer(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resizeto672(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resizeto672(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resizeto672(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet(1, 2, bilinear=True,aux=False)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    elif model=='transformer':
        model=ViT_seg(img_size=672,num_classes=2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    print(total_samples)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        # print(sampled_batch['image'].size())
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()

        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
        
        
        
        
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            # 将结果画出来
            if(save_result_img):
                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((denormalize(input_img,mean=mean,std=std).numpy()[j])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt=imgt.resize((740*3,350),Image.Resampling.LANCZOS)
                imgt.save(os.path.join(save_compair_image_file,file_path))
                presave = Image.fromarray(pred_img.astype(np.uint8))
                presave=presave.resize((740,350),Image.Resampling.LANCZOS)
                presavepath = os.path.join(dir_path,str(fold_num)+'/seg')
                if not os.path.exists(presavepath):
                    os.makedirs(presavepath)
                    
                presave.save(os.path.join(presavepath, sampled_batch["idx"][j][:-4] + '.jpg'))
                
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')

    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc,flops_g,elapsed_time,Mnum_params]
    # return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]

def cal_zhibiao_CNN_Transformer_Xpercent(dir_path,img_path,data_file,model_path,fold_num,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resizeto672_Xpercent(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resizeto672_Xpercent(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua_resizeto672_Xpercent(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet(1, 2, bilinear=True,aux=False)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    elif model=='transformer':
        model=ViT_seg(img_size=672,num_classes=2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    print(total_samples)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        # print(sampled_batch['image'].size())
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()

        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
        
        
        
        
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            # 将结果画出来
            if(save_result_img):
                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((denormalize(input_img,mean=mean,std=std).numpy()[j])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')

    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc,flops_g,elapsed_time,Mnum_params]
    # return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]


# brbs unet,单通道 简单标准化
def cal_zhibiao_hys_dantongdao(dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdao(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdao(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdao(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet_seg_2d(1, 2)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())

    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]

# 原本的网络,单通道,简单标准化
def cal_zhibiao_dantongdao_jiandanbiaozhunhua(dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdao(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdao(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdao(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            # model = UNet_seg_2d(1, 2)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())

    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
# brbs unet  单通道 特殊标准化
def cal_zhibiao_hys_dantongdao_teshubiaozhunhua(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:1'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet_seg_2d(1, 2)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
                    
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            # 将结果画出来
            if(save_result_img):
                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((denormalize(input_img,mean=mean,std=std).numpy()[j])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
                
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
def cal_zhibiao_hys_dantongdao_teshubiaozhunhua_RCPS(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:1'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet_seg_2d_RCPS_base(1, 2)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
                    
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            # 将结果画出来
            if(save_result_img):
                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((denormalize(input_img,mean=mean,std=std).numpy()[j])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
                presave = Image.fromarray(pred_img.astype(np.uint8))
                presavepath = os.path.join(dir_path,str(fold_num)+'/seg')
                if not os.path.exists(presavepath):
                    os.makedirs(presavepath)
                presave.save(os.path.join(presavepath, sampled_batch["idx"][j][:-4] + '.jpg'))
                
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
# 单通道简单归一化
def cal_zhibiao_hys_dantongdao_jiandanguiyihua_RCPS(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    # 加载数据集
    if data_type == 'val':
        dataset = hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path=img_path, split="val")
    elif data_type == 'test':
        dataset = hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path=img_path, split="test")
    elif data_type == 'train':
        dataset = hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path=img_path, split="train")
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")
    
    # 优化DataLoader参数以减少内存占用
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1,  # 减少工作线程数，降低内存消耗
        pin_memory=False,  # 禁用pin_memory以减少CPU到GPU的内存复制
        drop_last=False
    )
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            # model = UNet_seg_2d_RCPS_base(1, 2)
            model = UNet_seg_2d_RCPS_with_confidence(1, 2)
    elif model=='ABCnet_gn':
        model = ABCnet_gn_with_confidence(1, 2)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()
    
    # 如果使用GPU，启用cudnn benchmark以优化推理速度（可选）
    if torch.cuda.is_available() and device != 'cpu':
        torch.backends.cudnn.benchmark = True

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    
    # 初始化峰值显存记录
    peak_memory = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"初始显存占用: {initial_memory:.4f}MB")
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # 执行前向传播并获取输出
        with torch.no_grad():
            output = model(volume_batch)["output"]
        
        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
                    
        # 将输出转换为numpy数组并释放GPU内存
        output_np = output.cpu().detach().numpy().copy()
        output_np = np.argmax(output_np, axis=1)
        
        # 将标签转换为numpy数组并释放GPU内存
        label_batch_np = label_batch.cpu().detach().numpy().copy()
        
        # 释放不再需要的GPU张量
        del volume_batch, label_batch, output
        torch.cuda.empty_cache()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img = img.astype(np.uint8)
            
            # 连通区域分析和小区域过滤（优化版）
            output2 = measure.label(img[:, :], connectivity=2)
            
            # 使用numpy的bincount更高效地计算连通区域大小
            region_sizes = np.bincount(output2.flatten())
            
            # 过滤小区域
            small_regions = np.where(region_sizes < 100)[0]
            small_regions = np.append(small_regions, 0)  # 包括背景
            
            # 使用np.isin更高效地过滤小区域
            output2 = np.where(np.isin(output2, small_regions), 0, 1)
            
            # 更新混淆矩阵并计算指标
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int), img_label))
            recall_list.append(medpy.metric.binary.recall(output2, img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2, img_label))
            dc_list.append(medpy.metric.binary.dc(output2, img_label))
            jc_list.append(medpy.metric.binary.jc(output2, img_label))
            acc_list.append(matrix.acc_global())
            
            # 检查当前显存使用情况
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated() / 1024**2  # 转换为MB
                if current_memory > peak_memory:
                    peak_memory = current_memory
                print(f" 显存峰值: {peak_memory:.4f}MB")
            
            # 将结果画出来
            if(save_result_img):
                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((input_img.numpy()[j][0])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                
                # 创建对比图像
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                
                # 保存图像
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file, file_path))
                presave = Image.fromarray(pred_img.astype(np.uint8))
                presavepath = os.path.join(dir_path,str(fold_num)+'/seg')
                if not os.path.exists(presavepath):
                    os.makedirs(presavepath)
                presave.save(os.path.join(presavepath, sampled_batch["idx"][j][:-4] + '.jpg'))
        
        # 释放numpy数组占用的内存
        del output_np, label_batch_np
        import gc
        gc.collect()

    # 计算平均指标
    avg_iou = sum(iou_list) / len(iou_list)
    avg_hd95 = sum(hd95_list) / len(hd95_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_sensitivity = sum(sensitivity_list) / len(sensitivity_list)
    avg_dc = sum(dc_list) / len(dc_list)
    avg_jc = sum(jc_list) / len(jc_list)
    avg_acc = sum(acc_list) / len(acc_list)
    
    # 将结果写入文件
    file.write(f"{data_type}_list:\n")
    file.write(f"共{len(iou_list)}张{data_type}集,IOU:{avg_iou},HD95:{avg_hd95},recall:{avg_recall},sensitivity:{avg_sensitivity},dc:{avg_dc},jc:{avg_jc},acc:{avg_acc}\n")
    print(f"共{len(iou_list)}张{data_type}集,IOU:{avg_iou},HD95:{avg_hd95},recall:{avg_recall},sensitivity:{avg_sensitivity},dc:{avg_dc},jc:{avg_jc},acc:{avg_acc}")
    file.write(f"共{len(iou_list)}张{data_type}集,IOU:{avg_iou},recall:{avg_recall},sensitivity:{avg_sensitivity},dc:{avg_dc},jc:{avg_jc},acc:{avg_acc}\n")
    print(f"共{len(iou_list)}张{data_type}集,IOU:{avg_iou},recall:{avg_recall},sensitivity:{avg_sensitivity},dc:{avg_dc},jc:{avg_jc},acc:{avg_acc}")
    
    # 记录峰值显存信息
    if torch.cuda.is_available():
        final_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"最终显存占用: {final_memory:.4f}MB")
        print(f"峰值显存占用: {peak_memory:.4f}MB")
        file.write(f"峰值显存占用: {peak_memory:.4f}MB\n")
    
    # 关闭文件
    file.close()
    
    # 释放所有不再需要的变量
    del iou_list, hd95_list, recall_list, sensitivity_list, dc_list, jc_list, acc_list
    del dataloader, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return [avg_iou, avg_hd95, avg_recall, avg_sensitivity, avg_dc, avg_jc, avg_acc]

# 单通道简单归一化,包含对见过病人但未参与训练的病人数据进行测试
def cal_zhibiao_hys_dantongdao_jiandanguiyihua_RCPS_33percent(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    # if data_type=='val':
    #     # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
    #     dataset=hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSets_dantongdao_jiandanguiyihua_33percent(data_file=data_file, img_path=img_path,split="test")
    # if data_type=='train':
    #     # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
    #     dataset=hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            # model = UNet_seg_2d_RCPS_base(1, 2)
            model = UNet_seg_2d_RCPS_with_confidence(1, 2)
    elif model=='ABCnet_gn':
        model = ABCnet_gn_with_confidence(1, 2)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    seen_but_no_train_num= 0
    total_samples = len(dataloader.dataset)
    # print(total_samples)
    for i_batch, sampled_batch in enumerate(dataloader):
        seen_but_no_train_num= sampled_batch['seen_but_no_train_num']
        # print(seen_but_no_train_num)
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
                    
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            # 将结果画出来
            if(save_result_img):
                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((input_img.numpy()[j][0])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
                
    # seen_but_no_train
    sbnt_iou_list = iou_list[:seen_but_no_train_num]
    sbnt_hd95_list = hd95_list[:seen_but_no_train_num]
    sbnt_recall_list = recall_list[:seen_but_no_train_num]
    sbnt_sensitivity_list = sensitivity_list[:seen_but_no_train_num]
    sbnt_dc_list = dc_list[:seen_but_no_train_num]
    sbnt_jc_list = jc_list[:seen_but_no_train_num]
    sbnt_acc_list = acc_list[:seen_but_no_train_num]
    
    # no_seen
    ns_iou_list = iou_list[seen_but_no_train_num:]
    ns_hd95_list = hd95_list[seen_but_no_train_num:]
    ns_recall_list = recall_list[seen_but_no_train_num:]
    ns_sensitivity_list = sensitivity_list[seen_but_no_train_num:]
    ns_dc_list = dc_list[seen_but_no_train_num:]
    ns_jc_list = jc_list[seen_but_no_train_num:]
    ns_acc_list = acc_list[seen_but_no_train_num:]
    
    
    sbnt_avg_iou=sum(sbnt_iou_list)/len(sbnt_iou_list)
    sbnt_avg_hd95=sum(sbnt_hd95_list)/len(sbnt_hd95_list)
    sbnt_avg_recall=sum(sbnt_recall_list)/len(sbnt_recall_list)
    sbnt_avg_sensitivity=sum(sbnt_sensitivity_list)/len(sbnt_sensitivity_list)
    sbnt_avg_dc=sum(sbnt_dc_list)/len(sbnt_dc_list)
    sbnt_avg_jc=sum(sbnt_jc_list)/len(sbnt_jc_list)
    sbnt_avg_acc=sum(sbnt_acc_list)/len(sbnt_acc_list)
    

    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(sbnt_iou_list),data_type,sbnt_avg_iou,sbnt_avg_hd95,sbnt_avg_recall,sbnt_avg_sensitivity,sbnt_avg_dc,sbnt_avg_jc,sbnt_avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(sbnt_iou_list),data_type,sbnt_avg_iou,sbnt_avg_hd95,sbnt_avg_recall,sbnt_avg_sensitivity,sbnt_avg_dc,sbnt_avg_jc,sbnt_avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(sbnt_iou_list),data_type,sbnt_avg_iou,sbnt_avg_recall,sbnt_avg_sensitivity,sbnt_avg_dc,sbnt_avg_jc,sbnt_avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(sbnt_iou_list),data_type,sbnt_avg_iou,sbnt_avg_recall,sbnt_avg_sensitivity,sbnt_avg_dc,sbnt_avg_jc,sbnt_avg_acc)+'\n')
    
    ns_avg_iou=sum(ns_iou_list)/len(ns_iou_list)
    ns_avg_hd95=sum(ns_hd95_list)/len(ns_hd95_list)
    ns_avg_recall=sum(ns_recall_list)/len(ns_recall_list)
    ns_avg_sensitivity=sum(ns_sensitivity_list)/len(ns_sensitivity_list)
    ns_avg_dc=sum(ns_dc_list)/len(ns_dc_list)
    ns_avg_jc=sum(ns_jc_list)/len(ns_jc_list)
    ns_avg_acc=sum(ns_acc_list)/len(ns_acc_list)
    

    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(ns_iou_list),data_type,ns_avg_iou,ns_avg_hd95,ns_avg_recall,ns_avg_sensitivity,ns_avg_dc,ns_avg_jc,ns_avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(ns_iou_list),data_type,ns_avg_iou,ns_avg_hd95,ns_avg_recall,ns_avg_sensitivity,ns_avg_dc,ns_avg_jc,ns_avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(ns_iou_list),data_type,ns_avg_iou,ns_avg_recall,ns_avg_sensitivity,ns_avg_dc,ns_avg_jc,ns_avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(ns_iou_list),data_type,ns_avg_iou,ns_avg_recall,ns_avg_sensitivity,ns_avg_dc,ns_avg_jc,ns_avg_acc)+'\n')
    
    
    
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [sbnt_avg_iou,sbnt_avg_hd95,sbnt_avg_recall,sbnt_avg_sensitivity,sbnt_avg_dc,sbnt_avg_jc,sbnt_avg_acc],[ns_avg_iou,ns_avg_hd95,ns_avg_recall,ns_avg_sensitivity,ns_avg_dc,ns_avg_jc,ns_avg_acc],[avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
def cal_zhibiao_hys_dantongdao_jiandanguiyihua_RCPS_CL(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSets_dantongdao_jiandanguiyihua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:1'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet_seg_2d_RCPS_with_confidence_and_CL(1, 2,project_dim=32)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
                    
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            # 将结果画出来
            if(save_result_img):
                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((input_img.numpy()[j][0])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
                
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
def cal_zhibiao_Vnet_2d_dantongdao_teshubiaozhunhua(dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            # model = UNet_seg_2d(1, 2)
            model = VNet2d(1, 2, normalization='batchnorm', has_dropout=False)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())

    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]

# 原本网络  单通道 特殊标准化
def cal_zhibiao_dantongdao_teshubiaozhunhua(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdao(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)

    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        
        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
                    
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            #将连通域面积小于100像素的删除不计入统计
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            
            
            # #  不将连通域面积小于100像素的删除不计入统计
            # matrix = ConfusionMatrix_new(2)
            # img1, img_label = output_np[j], label_batch_np[j]
            # output2 =img1
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            
            # 将结果画出来
            if(save_result_img):

                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((denormalize(input_img,mean=mean,std=std).numpy()[j])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
                presave = Image.fromarray(pred_img.astype(np.uint8))
                presavepath = os.path.join(dir_path,str(fold_num)+'/seg')
                if not os.path.exists(presavepath):
                    os.makedirs(presavepath)
                presave.save(os.path.join(presavepath, sampled_batch["idx"][j][:-4] + '.jpg'))
                
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]

# lf的基础unet  单通道 简单归一化
def cal_zhibiao_dantongdao_jiandanguiyihua_lfunet(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0,save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdao_quanjiandu(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdao_quanjiandu(data_file=data_file, img_path=img_path,split="test")
    # if data_type=='train':
    #     # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
    #     dataset=hysDataSetsdantongdao(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)

    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    total_samples = len(dataloader.dataset)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        
        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
                    
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            #将连通域面积小于100像素的删除不计入统计
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            
            
            # #  不将连通域面积小于100像素的删除不计入统计
            # matrix = ConfusionMatrix_new(2)
            # img1, img_label = output_np[j], label_batch_np[j]
            # output2 =img1
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())
            
            # 将结果画出来
            if(save_result_img):

                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((input_img.numpy()[j][0])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
                presave = Image.fromarray(pred_img.astype(np.uint8))
                presavepath = os.path.join(dir_path,str(fold_num)+'/seg')
                if not os.path.exists(presavepath):
                    os.makedirs(presavepath)
                presave.save(os.path.join(presavepath, sampled_batch["idx"][j][:-4] + '.jpg'))
                
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]

# brbs  unet网络  三通道 特殊标准化
def cal_zhibiao_3tongdao(dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSets(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSets(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSets(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet_seg_2d(3, 2)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())

    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
# brbs  unet网络  三通道 简单
def cal_zhibiao_3tongdao_jiandanguiyihua(dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSets3tongdaojiandanguiyihua(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSets3tongdaojiandanguiyihua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSets3tongdaojiandanguiyihua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        if aux: 
            model = UNet(3, 2, bilinear=True,aux=True,aux_locate=aux_locate)
        else:
            # model = UNet(1, 2, bilinear=True,aux=False,aux_locate=aux_locate)
            model = UNet_seg_2d(3, 2)
    elif model=='unet_sdi':
        model=UNet_sdi(3,2,bilinear=True,aux=True,aux_locate=aux_locate,)
        # model=ViT_seg(num_classes=2).cuda()
    elif model=="attention":
        model=AttU_Net(3,2)
    # print(model)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    avg_acc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)["output"]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))

            acc_list.append(matrix.acc_global())

    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
def cal_zhibiao_unimatch(dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',aux_locate=0):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=MGD_BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    model = UNet_unimatch(in_chns=3, class_num=2)  

    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            avg_iou=sum(iou_list)/len(iou_list)

            # hd95_list.append(medpy.metric.binary.hd95(output2.astype(int),img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))
    avg_iou=sum(iou_list)/len(iou_list)
    # avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc)+'\n')
    file.write("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc)+'\n')
    print("共%d张%s集,IOU:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_recall,avg_sensitivity,avg_dc,avg_jc)+'\n')
    
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,0,avg_recall,avg_sensitivity,avg_dc,avg_jc]



def cct_cal_zhibiao(dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet'):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="val")
        
    if data_type=='test':
        # dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        model=UNet_CCT(3,2)
    else:
        model=ViT_seg(num_classes=2).cuda()
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)[0]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            avg_iou=sum(iou_list)/len(iou_list)
            hd95_list.append(medpy.metric.binary.hd95(output2,img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc)+'\n')
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc]


def cct_cal_zhibiao_hys_dantongdao(fold_num,dir_path,img_path,data_file,model_path,aux,data_type,device='cpu',add_content='',model='unet',save_result_img=False):
    '''
    dir_path是txt保存位置
    img_path是输入图像的位置
    data_file是指数据集名字traindic文件
    model_path是模型路径
    data_type是指明要计算训练集还是验证集还是测试集的指标
    '''

    model_name=model_path
    model_name=model_name.split('/')[-1].split('.')[0]
    print(model_name)
    data_file_name=data_file
    data_file_name=data_file_name.split('/')[-1].split('.')[0]
    assert os.path.exists(dir_path), dir_path + " is not exist"
    file=open(os.path.join(dir_path,data_file_name+model_name+'_'+add_content+'.txt'),'w' )

    if data_type=='val':
        # dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="val")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="val")
        
    if data_type=='test':
        # dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="test")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        # dataset=BaseDataSets(data_file=data_file, img_path=img_path,split="train")
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:1'


    # encoder=UNet_encoder(3,2,True,True)
    # decoder=UNet_decoder_gate(2,True)
    # encoder.load_state_dict(torch.load(model1_path,map_location=device))
    # decoder.load_state_dict(torch.load(model2_path,map_location=device))
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    if model=='unet':
        model=UNet_CCT(1,2)
    else:
        # model=ViT_seg(num_classes=2).cuda()
        print("请定义vit_seg")
    model.load_state_dict(torch.load(model_path,map_location=device))
    model=model.to(device)
    model.eval()

    iou_list=[]
    avg_iou=0.0
    hd95_list=[]
    avg_hd95=0.0
    recall_list=[]
    avg_recall=0.0
    sensitivity_list=[]
    avg_sensitivity=0.0 
    dc_list=[]
    avg_dc=0.0
    jc_list=[]
    avg_jc=0.0
    acc_list=[]
    total_samples = len(dataloader.dataset)
    for i_batch, sampled_batch in enumerate(dataloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
        
        # output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        output = model(volume_batch)[0]
        
        output_np = output.cpu().detach().numpy().copy()  
        output_np = np.argmax(output_np, axis=1)
        label_batch_np=label_batch.cpu().detach().numpy().copy()
        

        if i_batch == total_samples - 1:
            # 使用 thop 库统计 FLOPs
            flops, params = profile(model, inputs=(volume_batch,))
            flops_g = flops / 1e9  # 转换为 GFLOPs
            print(f"GFLOPs: {flops_g}")
            Mnum_params = params / 1e6  # 转换为 M
            print(f"模型参数量: {Mnum_params}M")

            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 记录开始时间
                start_event.record()
                
                # 执行前向传播
                with torch.no_grad():
                    model(volume_batch)
                
                # 记录结束时间
                end_event.record()
                
                # 等待所有 CUDA 核心执行完毕
                torch.cuda.synchronize()
                
                # 计算时间
                elapsed_time = start_event.elapsed_time(end_event)
                print(f"运算时间: {elapsed_time:.3f} 毫秒")
        
        
        #------------计算val iou---------#
        for j in range(output_np.shape[0]):
            matrix = ConfusionMatrix_new(2)
            img, img_label = output_np[j], label_batch_np[j]
            img=img.astype(np.uint8)
            # img=cv2.resize(img, (704, 320))
            # img_label=cv2.resize(img_label, (704, 320))
            output2 = measure.label(img[:, :], connectivity=2)
            max_ = max(output2.flatten()) + 1
            # print(max)
            l = np.zeros(max_)
            for i in output2.flatten():
                l[i] += 1
            # l=sorted(l)
            a = [i for i in range(len(l)) if l[i] < 100]
            a.append(0)
            # print(l, a)
            for i in range(len(output2.ravel())):
                if output2.ravel()[i] in a:
                    output2.ravel()[i] = 0
                else:
                    output2.ravel()[i] = 1
            # matrix.update(img, img_label)
            matrix.update(output2, img_label)
            iou_list.append(matrix.iou())
            avg_iou=sum(iou_list)/len(iou_list)
            hd95_list.append(medpy.metric.binary.hd95(output2,img_label))
            recall_list.append(medpy.metric.binary.recall(output2,img_label))
            sensitivity_list.append(medpy.metric.binary.sensitivity(output2,img_label))
            dc_list.append(medpy.metric.binary.dc(output2,img_label))
            jc_list.append(medpy.metric.binary.jc(output2,img_label))
            
            
            acc_list.append(matrix.acc_global())
            # 将结果画出来
            if(save_result_img):

                file_path = sampled_batch["idx"][j][:-4]+"_compair.jpg"
                save_compair_image_file = os.path.join(dir_path,str(fold_num)+"/compair_segmentation_image_result")
                if not os.path.exists(save_compair_image_file):
                    os.makedirs(save_compair_image_file)
                input_img =sampled_batch['image'].cpu()

                input_img = ((denormalize(input_img,mean=mean,std=std).numpy()[j])*255).astype(np.uint8)
                lab = (img_label*255).astype(np.uint8)
                # input_img = cv2.resize(input_img,(output2.shape[1], output2.shape[0]))  # 根据预测图像进行resize
                pred_img = output2*255
                lab_img = lab
                img = np.zeros([input_img.shape[0], input_img.shape[1] * 3])
                img[:, :input_img.shape[1]] = input_img
                img[:, input_img.shape[1]:input_img.shape[1] * 2] = pred_img
                img[:, input_img.shape[1] * 2:] = lab_img
                imgt = Image.fromarray(img.astype(np.uint8))
                imgt.save(os.path.join(save_compair_image_file,file_path))
                presave = Image.fromarray(pred_img.astype(np.uint8))
                presavepath = os.path.join(dir_path,str(fold_num)+'/seg')
                if not os.path.exists(presavepath):
                    os.makedirs(presavepath)
                presave.save(os.path.join(presavepath, sampled_batch["idx"][j][:-4] + '.jpg'))
                
    avg_iou=sum(iou_list)/len(iou_list)
    avg_hd95=sum(hd95_list)/len(hd95_list)
    avg_recall=sum(recall_list)/len(recall_list)
    avg_sensitivity=sum(sensitivity_list)/len(sensitivity_list)
    avg_dc=sum(dc_list)/len(dc_list)
    avg_jc=sum(jc_list)/len(jc_list)
    avg_acc=sum(acc_list)/len(acc_list)
    file.write(data_type+'_list:\n')
    file.write("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
    print("共%d张%s集,IOU:%f,HD95:%f,recall:%f,sensitivity:%f,dc:%f,jc:%f,acc:%f"\
        %(len(iou_list),data_type,avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc)+'\n')
   
            
            
            
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # file.write("总参数数量和：" + str(k)+'\n')
    file.close()
    return [avg_iou,avg_hd95,avg_recall,avg_sensitivity,avg_dc,avg_jc,avg_acc]
if __name__=='__main__':
    cal_zhibiao('zhibiao/zhibiao_9_17_up&down_mgd_newdata_pseudo_lossis_mse',img_path="data_new_2/image_crop",data_file='data_new_2/train_dict1.pkl',model_path='./model_9_17_up&down_mgd_newdata_pseudo_lossis_mse/model1_best.pth',data_type="test",device='cpu')
