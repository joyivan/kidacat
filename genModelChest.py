import os

import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from mayavi import mlab
import maxTool
import cv2
import scipy
#ll
#baseDir='/media/joyivan/0a1457c2-893b-40d4-bd3f-a5316e4b4854/CT_MD/COVID-19_Cases/'
baseDir='/Users/lizirong/Downloads/CT_MD/COVID-19_Cases'
sampeDir=os.listdir(baseDir)
print(sampeDir)


def gen3DByDirName(base,Dirname,angle=0):
    fileList=os.listdir(os.path.join(base,Dirname))
    fileList=[i for i in fileList if i[-4:]=='.jpg']
    fileNumber=(len(fileList))
    chest3d=np.ones((fileNumber,512,512),dtype=np.uint8)

    for i in range(fileNumber):
            print('IM'+str(i+1).zfill(4)+'.jpg')
            pfile=cv2.imread(os.path.join(base,Dirname,'IM'+str(i+1).zfill(4)+'.jpg'),0)
            #print(pfile.shape)

            singleFile=cv2.imread(os.path.join(base,Dirname,'IM'+str(i+1).zfill(4)+'.jpg'),cv2.IMREAD_GRAYSCALE)
           # cv2.cvtColor(singleFile, cv2.COLOR_BGR2RGB, singleFile)
           # print(singleFile.dtype)
            #cv2.imshow('temp', singleFile)
            #cv2.waitKey(0)
            singleFile=maxTool.cutBed(singleFile)
            print(singleFile.shape)
            plt.imshow(singleFile,cmap='gray')
            singleFile=ndi.rotate(singleFile,angle,reshape=False)
            #plt.imshow(singleFile,cmap='gray')
            plt.show()
            #input()
            chest3d[i,:,:]=singleFile
    return chest3d




temp=gen3DByDirName(baseDir,'P001',0)
print(temp.shape)
temp=scipy.ndimage.zoom(temp,(3,1,1),output=None,order=3,mode='constant',cval=0.0,prefilter=True)
print(temp.shape)
#topView=np.max(temp,axis=0)
#sideView=np.max(temp,axis=2)
#frontView=np.max(temp,axis=1)
#plt.imshow(frontView,cmap='gray')
#plt.show()
np.save('p001.npz',temp)
mlab.contour3d(temp.transpose(1,2,0),transparent=True,colormap='accent')               #显示表面
'''
colormap value
accent       flag          hot      pubu     set2
autumn       gist_earth    hsv      pubugn   set3
black-white  gist_gray     jet      puor     spectral
blue-red     gist_heat     oranges  purd     spring
blues        gist_ncar     orrd     purples  summer
bone         gist_rainbow  paired   rdbu     winter
brbg         gist_stern    pastel1  rdgy     ylgnbu
bugn         gist_yarg     pastel2  rdpu     ylgn
bupu         gnbu          pink     rdylbu   ylorbr
cool         gray          piyg     rdylgn   ylorrd
copper       greens        prgn     reds
dark2        greys         prism    set1
infor:https://www.cnblogs.com/dalanjing/p/12289517.html
todo:mlab.points3d(a.transpose(1,2,0),colormap='Greys')   

'''
mlab.show()
#print(temp.shape)
#cv2.imshow('temp',temp[0])
#cv2.waitKey(0)
#plt.imshow(temp[0])
#plt.show()

#def RodriguesRotate(v:np.ndarray,u:np.ndarray,theta:float)->np.ndarray:
#    '''向量v绕向量u旋转角度θ,得到新的向量P_new
#    罗德里格斯旋转公式:v' = vcosθ + (u×v)sinθ + (u·v)u(1-cosθ)
#
#    args:
#        v:向量,维度为(3,)
#        u:作为旋转轴的向量,维度为(3,)
#        theta:旋转角度θ,此处默认为角度值
#    returns:
#        v_new:旋转后得到的向量,维度为(3,)
#    '''
#    u = u/np.linalg.norm(u) # 计算单位向量
#    sin_theta = np.sin(theta*np.pi/180)
#    cos_theta = np.cos(theta*np.pi/180)
#    v_new = v*cos_theta + np.cross(u,v)*sin_theta + np.dot(u,v)*u*(1-cos_theta)
#    return v_new
#
#chestResult=chestRotate.rotation(temp,0,90,c=np.array([]))
