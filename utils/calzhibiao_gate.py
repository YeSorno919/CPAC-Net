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
import medpy

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

def cal_zhibiao(dir_path,img_path,data_file,model1_path,model2_path,data_type,device='cpu',add_content='',aux=True):
# def cal_zhibiao(dir_path,img_path,data_file,model_path,data_type,device='cpu',add_content=''):
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


    encoder=UNet_encoder(3,2,True,aux)
    decoder=UNet_decoder_gate(2,True)
    encoder.load_state_dict(torch.load(model1_path,map_location=device))
    decoder.load_state_dict(torch.load(model2_path,map_location=device))
    encoder.to(device).eval()
    decoder.to(device).eval()

    # model = UNet(3, 2, bilinear=True,aux=True)
    # model.load_state_dict(torch.load(model_path,map_location=device))
    # model=model.to(device)
    # model.eval()

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
        # output = model(volume_batch)["output"]
        
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
def cal_zhibiao_hys(fold_num,dir_path,img_path,data_file,model1_path,model2_path,data_type,device='cpu',add_content='',aux=True,save_result_img=False):
# def cal_zhibiao(dir_path,img_path,data_file,model_path,data_type,device='cpu',add_content=''):
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
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="val")
    if data_type=='test':
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="test")
    if data_type=='train':
        dataset=hysDataSetsdantongdaoteshubiaozhunhua(data_file=data_file, img_path=img_path,split="train")
    dataloader=DataLoader(dataset, batch_size=1, shuffle=False,num_workers=1)
    if device!='cpu':
        device='cuda:0'


    encoder=UNet_encoder(1,2,True,aux)
    decoder=UNet_decoder_gate(2,True)
    encoder.load_state_dict(torch.load(model1_path,map_location=device))
    decoder.load_state_dict(torch.load(model2_path,map_location=device))
    encoder.to(device).eval()
    decoder.to(device).eval()

    # model = UNet(3, 2, bilinear=True,aux=True)
    # model.load_state_dict(torch.load(model_path,map_location=device))
    # model=model.to(device)
    # model.eval()

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
        
        output = decoder(volume_batch,encoder(volume_batch)["feature"])["output"]
        # output = model(volume_batch)["output"]
        
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
if __name__=='__main__':
    cal_zhibiao('zhibiao/zhibiao_9_17_up&down_mgd_newdata_pseudo_lossis_mse',img_path="data_new_2/image_crop",data_file='data_new_2/train_dict1.pkl',model_path='./model_9_17_up&down_mgd_newdata_pseudo_lossis_mse/model1_best.pth',data_type="test",device='cpu')
