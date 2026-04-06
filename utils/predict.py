import heapq
from skimage import measure
from unet_model import UNet
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from unet_model import UNet
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	# ,transforms.Resize((360,740))
])
# model_weight_path = r'E:\FCN-8s\pytorch-FCN-easiest-demo-master\model_11_29\fcn_model_90.pt'
# model = torch.load(model_weight_path)
# # model.load_state_dict(torch.load(model_weight_path))
# or_path = r'E:/meibomian_glands/data_new/img_cla/'
# label_path = r'E:/meibomian_glands/data_new/label_crop/'
# i=1
# img = cv2.imread(or_path + "{}.jpg".format(str(i)))
# img_label = Image.open(label_path + "%03d.gif" % i)
# img_label = np.array(img_label)
# # img=np.zeros((320,704,3))
# # print(img.shape)
# # img = np.transpose(img, [2, 0, 1])
# img=img.astype("float")
# # img_label = cv2.resize(img_label, (704, 320))
# img = transform(img)
# img = torch.unsqueeze(img, dim=0)
# # img = np.transpose(img, [2, 0, 1])
# print(img.shape)
# output = model(img.to(torch.float32))
# output=torch.sigmoid(output)
# output=output.detach().numpy()[0]
# print(output.shape)
# plt.subplot(121),plt.imshow(output[0],"gray")
# plt.subplot(122),plt.imshow(output[1],"gray")
# plt.show()
# im = np.squeeze(output['x5'].detach().numpy())
# im = np.transpose(im, [1, 2, 0])
# print(im.shape)
# for i in range(im.shape[2]):
# 	plt.subplot(4,8,i+1),plt.imshow(im[:,:,i],"gray")
# plt.subplot(121),plt.imshow(im[:,:,0],"gray")
# plt.subplot(122),plt.imshow(img_label,"gray")
# plt.show()

def predict(model_path,img_path,name,pseudo_color=False):
	#单张图片预测
	#使用FCN预测输出shape为(320,704)
	# model_weight_path = r'E:\FCN-8s\pytorch-FCN-easiest-demo-master\model_2_20\fcn_model_100.pt'
	model_weight_path = model_path
	model = UNet(3, 2, bilinear=True)
	model.load_state_dict(torch.load(model_weight_path,map_location=torch.device('cpu')))
	# model.eval()
	i=name
	or_path = img_path
	label_path = r'data_new_2/label_crop/'
	img = cv2.imread(os.path.join(or_path , "%03d.png"%i))
	img_label = Image.open(label_path + "%03d.gif" % i)
	img_label = np.array(img_label)
	# img_label=cv2.resize(img_label,(704,320))
	img = cv2.resize(img, (704, 320))
	if pseudo_color:
		img=cv2.applyColorMap(img,3)
	# img = np.transpose(img, [2, 0, 1])
	img = transform(img)
	img = torch.unsqueeze(img, 0)
	# print(img.shape)
	# print(img.shape)
	output = model(img)
	# print(output.shape)
	output = torch.softmax(output,dim=1)
	output = output.detach().numpy()


	# output=output.numpy()
	# print(output.shape)
	output = np.argmax(output, axis=1)[0]#output 值为0和1  #因为没有使用onehot，所以注释掉
	# —————————————距离变换加——————————
	# output=output[0][0]#不为1类时删除
	#
	# for i in range(len(output.ravel())):
	# 	if output.ravel()[i] < 0.1:
	# 		output.ravel()[i] = 0
	# 	# print(output1.ravel()[i])
	# 	else:
	# 		output.ravel()[i] = 1
	# —————————————距离变换加——————————


	# output=np.array(output)

	# print(type(output))
	output=output.astype(np.uint8)
	output = cv2.resize(output, (740, 350), interpolation=cv2.INTER_LINEAR_EXACT)


	# output2=output.copy()
	# output2=cv2.cvtColor(output2,cv2.COLOR_BAYER_GR2GRAY)
	# contours, hierarchy = cv2.findContours(output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# area = []
	# for i in range(len(contours)):
	# 	area.append(cv2.contourArea(contours[i]))
	# for i in range(len(area)):
	# 	if area[i] <= 50:
	# 		output=cv2.fillConvexPoly(output, contours[i], 0)
	# for i in output.flatten():
	# 	if i !=0:
	# 		print(i)
	#————————后处理  output2为后处理之后结果————————#
	output2 = measure.label(output[:, :], connectivity=2)
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

	# ————————后处理————————#
	return output2,img_label,model

def more_save(model_path,image_path,label_path,save_path):
	#多张图片预测后保存
	#使用FCN预测输出shape为(320,704)
	# model_weight_path = r'E:\FCN-8s\pytorch-FCN-easiest-demo-master\model_12_3\fcn_model_100.pt'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	model_weight_path = model_path
	model = torch.load(model_weight_path,map_location=torch.device('cpu'))
	# for i in range(1, 194):
	for file in os.listdir(image_path):
		file_num=int(os.path.splitext(file)[0])
		img = cv2.imread(os.path.join(image_path , "{:03d}.jpg".format(file_num)))
		img_label = Image.open(os.path.join(label_path ,"%03d.gif" % file_num))
		img_label = np.array(img_label)
		img = cv2.resize(img, (704, 320))
		# img = np.transpose(img, [2, 0, 1])
		img = transform(img)
		img = torch.unsqueeze(img, 0)
		# print(img.shape)
		output = model(img)
		# print(output.shape)
		output = torch.sigmoid(output)
		output = output.detach().numpy()
		print(file_num)
		# output = np.argmin(output, axis=1)[0]#shape为(320,704)#因为没有使用onehot，所以注释掉
		#———————————————距离变换——————————————#
		output=output[0][0]#不为1类时删除
		for j in range(len(output.ravel())):
			if output.ravel()[j] < 0.1:
				output.ravel()[j] = 0
			# print(output1.ravel()[i])
			else:
				output.ravel()[j] = 1
		output = output.astype("int64")
		# ———————————————距离变换——————————————#
		output = cv2.resize(output, (740, 350), interpolation=cv2.INTER_LINEAR_EXACT)

		# ————————后处理  output2为后处理之后结果————————#
		output2 = measure.label(output[:, :], connectivity=2)
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

		# ————————后处理————————#

		cv2.imwrite(os.path.join(save_path,"{}.jpg".format(str(file_num))),output2*255)
		# plt.imsave(os.path.join(save_path,"{}.gif".format(str(i))),output*255,cmap="gray")
		# plt.subplot(121),plt.imshow(output,"gray")
		# plt.subplot(122), plt.imshow(img_label, "gray")
		# plt.show()
		# for i in (output*255).flatten():
		# 	print(i)
		# break
	return output,img_label


if __name__=="__main__":
	#——————————————————————————使用more_save——————————————————————#
	# more_save( r'model_3_24/fcn_model_140.pt',r'meibomian_glands_new/img_cla/', r'meibomian_glands_new/label_crop/', r'meibomian_glands_new/pic_3_24_140_contour/')
	#——————————————————————————使用more_save——————————————————————#
	# output,label=predict(r'model_3_24/fcn_model_140.pt',47)
	# # for i in output.flatten():
	# # 	if i >0.1:
	# # 		print(i)
	# # print(output.shape)
	# # output=output.detach().numpy()[0]
	# output1=output.copy()
	# # # print(output.shape)
	# # # label=cv2.distanceTransform(label, cv2.DIST_L1, 3)
	# # output2=output1.copy()
	# for i in range(len(output1.ravel())):
	# 	if output1.ravel()[i]<0.8:
	# 		output1.ravel()[i]=0
	# 		# print(output1.ravel()[i])
	# 	else :
	# 		output1.ravel()[i] = 1
	# # # for i in output1.flatten():
	# # # 	print(i)
	# plt.subplot(221),plt.imshow(output,"gray")
	# plt.subplot(222),plt.imshow(label,"gray")
	# plt.subplot(223), plt.imshow(output1, "gray")
	# plt.show()
	# for i in output1.flatten():
	# 	if i>0:
	# 		print(i)
	# # output2=output.copy()
	# # for i in range(len(output.ravel())):
	# # 	if output.ravel()[i]<0.9:
	# # 		output.ravel()[i]=0
	#
	#
	# print(output[0].shape)
	# print("1")
	#—————————————————————————使用predict—————————————————————————#
	model_path = r'model_4_2/fcn_model_150.pt'
	img = cv2.imread("test_1.jpg")
	model = torch.load(model_path, map_location=torch.device('cpu'))
	img = cv2.resize(img, (704, 320))
	img = transform(img)
	img = torch.unsqueeze(img, 0)
	output = model(img)
	# print(output.shape)
	output = torch.sigmoid(output)
	output = output.detach().numpy()
	output = np.argmin(output, axis=1)[0]#output 值为0和1  #因为没有使用onehot，所以注释掉
	# print(img.shape)
	output = output.astype(np.uint8)
	# output = cv2.resize(output, (740, 350), interpolation=cv2.INTER_LINEAR_EXACT)
	output2 = measure.label(output[:, :], connectivity=2)
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
	plt.subplot(111),plt.imshow(output2,"gray"),plt.show()
	#—————————————————————————使用predict—————————————————————————#
	#——————————————————使用more_save批量保存——————————————#
	# more_save(r'E:\FCN-8s\pytorch-FCN-easiest-demo-master\model_12_1\fcn_model_100.pt',r"E:\meibomian_glands\data_new\img_cla",
	# 		  "E:\meibomian_glands\data_new\label_crop",r"E:\meibomian_glands\data_new\result_12_1")
	#——————————————————使用more_save批量保存——————————————#

	#——————————————————将预测图象保存————————————————#
	# path="E:/meibomian_glands/data_new/result_12_1/"
	# for i in range(1,194):
	# 	a,b=predict(i)
		# cv2.imwrite(path+str(i)+".jpg",a*255)
	# a,b=predict(1)
	# plt.subplot(1,2,1),plt.imshow(a,"gray")
	# plt.subplot(1, 2, 2), plt.imshow(b, "gray")
	# plt.show()
	#——————————————————将预测图象保存————————————————#
	#————————————测试sigmoid之后阈值的选择 还是np.argmin()最好但是不知道原理。使用这个的时候把predict的out=np.argmin()注释掉——————#
	# a,b=predict(1)
	# c=np.argmin(a, axis=1)[0]
	# a=a[0][0]
	# num_0=0.0
	# num_1=0.0
	# for i in b.flatten():
	# 	if i==0:
	# 		num_0+=1
	# 	if i==1:
	# 		num_1+=1
	# bi=num_1/num_0
	# print(bi)
	# a=cv2.threshold(a,bi,1,cv2.THRESH_BINARY)[1]
	# plt.subplot(121),plt.imshow(a,"gray")
	# plt.subplot(122),plt.imshow(c,"gray")
	# plt.show()
	#————————————测试结束——————————————————————————————————————————————
	# model_weight_path = r'E:\FCN-8s\pytorch-FCN-easiest-demo-master\model_11_29\fcn_model_100.pt'
	# model = torch.load(model_weight_path)
	# or_path = r'E:/meibomian_glands/data_new/img_cla/'
	# label_path = r'E:/meibomian_glands/data_new/label_crop/'
	# i=1
	# img = cv2.imread(or_path + "{}.jpg".format(str(i)))
	# img_label = Image.open(label_path + "%03d.gif" % i)
	# img_label = np.array(img_label)
	# img=cv2.resize(img,(704,320))
	# # img = np.transpose(img, [2, 0, 1])
	# img=transform(img)
	# img=torch.unsqueeze(img,0)
	# # print(img.shape)
	# output=model(img)
	# # print(output.shape)
	# output=torch.sigmoid(output)
	# output=output.detach().numpy()
	# print(output.shape)
	# output = np.argmin(output, axis=1)[0]
	# # print(output.shape)
	# # for i in range(output.shape[0]):
	# # 	for j in range(output.shape[1]):
	# # 		if output[i][j]<=0.4:
	# # 			output[i][j]=0
	# # 		else:
	# # 			output[i][j]=1
	
	# print(output.shape)
	# plt.subplot(121),plt.imshow(output,"gray")
	# # # plt.subplot(122),plt.imshow(output[1],"gray")
	# plt.show()
