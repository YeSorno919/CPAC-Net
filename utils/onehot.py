import numpy as np
import cv2
from PIL import Image
def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    # img_onehot=buf*255
    # print("buf.shape",buf.shape)
    # img_onehot=np.transpose(buf,(2,0,1))
    # img_onehot[0]=cv2.threshold(img_onehot[0],200,255,cv2.THRESH_BINARY)[1]
    # img_onehot[1]=cv2.threshold(img_onehot[1],200,255,cv2.THRESH_BINARY)[1]

    # kernel=np.ones((5,5),np.uint8)
    # img_onehot[0]=cv2.morphologyEx(img_onehot[0],cv2.MORPH_CLOSE,kernel)
    # img_onehot[1]=cv2.morphologyEx(img_onehot[1],cv2.MORPH_OPEN,kernel)
    # print("nmsk.shape",buf.shape)
    
    # return img_onehot
    return buf

if __name__=="__main__":
    im=Image.open(r"E:/meibomian_glands/data_new/label_crop/001.gif")
    im=np.array(im)
    # im=cv2.imread("E:/meibomian_glands/data_new/label_crop/001.gif",0)
    # im=cv2.threshold(im,200,255,cv2.THRESH_BINARY)[1]
    cv2.imshow("",im*255)
    cv2.waitKey(0)
    im_onehot=onehot(im,2)
    print(im.shape)

    cv2.imshow("1",im_onehot[:,:,0])
    cv2.imshow("2",im_onehot[:,:,1])
    cv2.waitKey(0)
    print(im_onehot.shape)


# if __name__=='__main__':
#     pass
    # img = cv2.imread('E:/meibomian_glands/data_new/label_crop/99.jpg', 0)
    # img=cv2.threshold(img,120,255,cv2.THRESH_BINARY)[1]
    # for i in img.ravel():
    #     if i!=0 and i!=255:
    #         print(i)
    # im=Image.open('E:/meibomian_glands/data_new/label_crop/001.gif')
    # im=np.array(im)
    # # img=img/255
    # im=im.astype('uint8')
    # img_onehot=onehot(im,2)
    # img_onehot=img_onehot.transpose(2,0,1)
    # for i in (im==img_onehot[0]).flatten():
    #     if i == False:
    #         print(i)
    # print("img.shape:",img.shape)
    # n=2
    # print(img)
    # print("img.size:",img.size)
    # buf=np.zeros(img.shape+(n,))
    # print("(n,):",(n,))
    # print("(350,740)+(2,1):",(350,740)+(2,))#元组相加相当于升维
    # print("img.shape+(n,):",img.shape+(n,))
    # print("buf:",buf)
    # print("np.arange*n:",np.arange(img.size)*n)
    # print("img.ravel()",img.ravel())#将img展开为1维



    # img_onehot=onehot(img,2)
    # img_onehot*=255
    # img_onehot = img_onehot.transpose(2,0,1)[0]
    # print("img_onehot.sahpe:",img_onehot)

    # print(img_onehot.shape)
    # img_onehot=np.transpose(img_onehot,(2,0,1))
    # kernel1=np.ones((3,3),np.uint8)
    # kernel2=np.zeros((11,11),np.uint8)
    # img_onehot[0]=cv2.morphologyEx(img_onehot[0],cv2.MORPH_CLOSE,kernel1)
    # img_onehot[1]=cv2.morphologyEx(img_onehot[1],cv2.MORPH_OPEN,kernel1)
    # img_onehot[0]=cv2.threshold(img_onehot[0],200,255,cv2.THRESH_BINARY)[1]
    # img3=img2/255
    # img3=img3.astype('uint8')
    # img3=img3*255
    # print(img_onehot)
    # cv2.imshow("1",img_onehot[0])
    # cv2.imshow("2",img_onehot[1])
    # # cv2.imshow("3",img3)
    # cv2.imshow("1",img_onehot[0])
    # cv2.waitKey(0)    
