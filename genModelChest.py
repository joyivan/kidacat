import os
import pydicom
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
from mayavi import mlab
import maxTool
import cv2
import scipy
#ll
#baseDir='/media/joyivan/0a1457c2-893b-40d4-bd3f-a5316e4b4854/CT_MD/COVID-19_Cases/'
#baseDir='/home/joyivan/Downloads/data/CT_MD/COVID-19_Cases'
baseDir='/Users/lizirong/Downloads/CT_MD/COVID-19_Cases'
sampeDir=os.listdir(baseDir)
print(sampeDir)


def gen3DByDirName(base,Dirname,angle=90):
    fileList=os.listdir(os.path.join(base,Dirname))
    fileList=[i for i in fileList if i[-4:]=='.jpg']
    fileNumber=(len(fileList))
    chest3d=np.ones((fileNumber,512,512),dtype=np.uint8)

    for i in range(fileNumber):
            print('IM'+str(i+1).zfill(4)+'.jpg')
            #pfile=cv2.imread(os.path.join(base,Dirname,'IM'+str(i+1).zfill(4)+'.jpg'),0)
            #print(pfile.shape)

            singleFile=cv2.imread(os.path.join(base,Dirname,'IM'+str(i+1).zfill(4)+'.jpg'),cv2.IMREAD_GRAYSCALE)
           # cv2.cvtColor(singleFile, cv2.COLOR_BGR2RGB, singleFile)
           # print(singleFile.dtype)
            #cv2.imshow('temp', singleFile)
            #cv2.waitKey(0)
            #print(singleFile.shape)
            singleFile=maxTool.cutBed(singleFile)
            print(singleFile.shape)
            #plt.imshow(singleFile,cmap='gray')
            #plt.show()
            singleFile=ndi.rotate(singleFile,angle,reshape=False)
            #plt.imshow(singleFile,cmap='gray')
            #plt.show()
            #input()
            chest3d[i,:,:]=singleFile
    return chest3d




#temp=gen3DByDirName(baseDir,'P001',0)
#print(temp.shape)
#temp=scipy.ndimage.zoom(temp,(3,1,1),output=None,order=3,mode='constant',cval=0.0,prefilter=True)
#print(temp.shape)
#topView=np.max(temp,axis=0)
#sideView=np.max(temp,axis=2)
#frontView=np.max(temp,axis=1)
#plt.imshow(frontView,cmap='gray')
#plt.show()
#np.save('p001.npz',temp)
#mlab.contour3d(temp.transpose(1,2,0),transparent=True)               #显示表面
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
#mlab.show()

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
def gen3DByDirNameDicom(base,Dirname,angle=0):
    fileList=os.listdir(os.path.join(base,Dirname))
    fileList=[i for i in fileList if i[-4:]=='.dcm']
    fileNumber=(len(fileList))
    chest3d=np.zeros((fileNumber,512,512),dtype=np.uint16)

    for i in range(fileNumber):
        singleFile=pydicom.dcmread(os.path.join(base,Dirname,'IM'+str(i+1).zfill(4)+'.dcm')).pixel_array
        print(singleFile.shape)
        chest3d[i,:,:]=singleFile
        print('method max',chest3d.max())
    return chest3d
'''
from lungmask import LMInferer
temp=gen3DByDirNameDicom(baseDir,'P001',0)
#temp=np.transpose(temp,(1,2,0))
print('max:',temp.max())
print('min:',temp.min())
print(temp.shape)
temp=temp.transpose(1,2,0)
inferer = LMInferer(force_cpu=False)
segmentation = inferer.apply(temp)
print('max:',segmentation.max())
print('min:',segmentation.min())

print('non zero counts:', np.count_nonzero(segmentation))
#segmentation[np.nonzero(segmentation)]=255
np.save('p001SEG',segmentation)
mlab.contour3d(segmentation,transparent=True)               #显示表面
#mlab.contour3d(temp.transpose(1,2,0),transparent=True)               #显示表面
mlab.show()
'''

def segByNii(file):
    import nibabel as nb
    '''load nii后仔axis2进行zoom，然后对整体进行lungmask预测，原数据data值仔-1099---+1899（约）内
    所以减去最小值，除以255，分割后的所以非零位置赋值255，矩阵点乘，contour3d作图'''
    img = nb.load(file)  # 读取nii格式文件
    img_affine = img.affine
    data = np.asanyarray(img.dataobj)
    data=scipy.ndimage.zoom(data,(1,1,3),output=None,order=3,mode='constant',cval=0.0,prefilter=True)

    print(data.shape)
    from lungmask import LMInferer
    inferer = LMInferer(force_cpu=False)
    segmentation = inferer.apply(data)
    print('max:', segmentation.max())
    print('min:', segmentation.min())
    #np.save('dataSeg',(data,segmentation))
    segmentation[np.nonzero(segmentation)]=255
    result=np.multiply((data-data.min())/255,segmentation)
    return (data,segmentation,result)


def segByNii2(file):
    import nibabel as nb
    '''load nii后仔axis2进行zoom，然后对整体进行图像学预测，原数据data值仔-1099---+1899（约）内
    所以减去最小值，除以255，分割后的所以非零位置赋值255，矩阵点乘，contour3d作图'''
    img = nb.load(file)  # 读取nii格式文件
    img_affine = img.affine
    data = np.asanyarray(img.dataobj)
    data=scipy.ndimage.zoom(data,(1,1,3),output=None,order=3,mode='constant',cval=0.0,prefilter=True)

    print(data.shape)

    from lungmask import LMInferer
    inferer = LMInferer(force_cpu=False)
    segmentation = inferer.apply(data)

    print('max:', segmentation.max())
    print('min:', segmentation.min())
    #np.save('dataSeg',(data,segmentation))
    segmentation[np.nonzero(segmentation)]=255
    result=np.multiply((data-data.min())/255,segmentation)
    return (data,segmentation,result)
#(data,segResult)=segByNii(baseDir+'/P001/1.nii.gz')

''''
get dcm to nii and use nii to predict using lungmask
pip install dicom2nifti 
import dicom2nifti
 

dicom2nifti.convert_directory(path_of_dycom_series,path_of_nii_file)
#第一个参数是dycom文件的目录，第二个是你要保存nii文件的目录。需要注意的是，默认输出的是nii.gz压缩文件，你也可以通过compression=False,选择不压缩nii文件

import nibabel as  nb

img = nb.load(xxx.nii.gz) #读取nii格式文件
img_affine = img.affine
data=np.asanyarray(img.dataobj)
#data.shape
#(512, 512, 148)
from lungmask import LMInferer
inferer = LMInferer(force_cpu=False)
lungmask 2023-10-29 23:49:33 No GPU found, using CPU instead
segmentation = inferer.apply(data)
from mayavi import mlab
mlab.contour3d(segmentation,transparent=True)    
'''
def sortSlice(dirpath):
    '''

    根据病人pydicom读入的SliceLocation对数据重新排序,因为github数据作者说文件顺序不一定是slice顺序
    :param dirpath:
    病人文件夹
    :return:返回darry [512 512 slicenumber]
    '''
    fileList=os.listdir(dirpath)

    print(dirpath)
    fileList=[s for s in fileList if s[-4:]=='.dcm']
    temp=dict()
    print(fileList)
    for i in fileList:
       file=pydicom.dcmread(os.path.join(dirpath,i))
       temp.update({file.SliceLocation:i})
    print(temp.keys())
    soredSlice=sorted(temp.keys())
    print('sorted:',soredSlice)

    result=np.zeros((512,512,len(fileList)))
    indexF=0
    for i in soredSlice:
       file=pydicom.dcmread(os.path.join(dirpath,temp[i]))
       fileData=file.pixel_array*int(file.RescaleSlope) + int(file.RescaleIntercept)

       result[:,:,indexF]=fileData
       indexF+=1
    result=scipy.ndimage.zoom(result,(1,1,3),output=None,order=3,mode='constant',cval=0.0,prefilter=True)
    return result

result=sortSlice(baseDir+'/P001/')
print(result.shape)
#mlab.contour3d(result,transparent=True)
#mlab.show()
def getAndPlotLung(lungData):
        '''infer leng senment,plot 3d lung and  return lung data'''
        from lungmask import LMInferer
        inferer = LMInferer(force_cpu=False)
        segmentation = inferer.apply(lungData)

        print('max:', segmentation.max())
        print('min:', segmentation.min())
        #np.save('dataSeg',(data,segmentation))
        segmentation[not np.nonzero(segmentation)]=segmentation.min()
        result=np.multiply(lungData,segmentation)

        mlab.contour3d(result,transparent=True)
        mlab.show()
        return result
getAndPlotLung(result)