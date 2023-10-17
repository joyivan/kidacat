import os
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
baseDir='/media/joyivan/0a1457c2-893b-40d4-bd3f-a5316e4b4854/CT_MD/COVID-19_Cases/'
sampeDir=os.listdir(baseDir)
print(sampeDir)


def gen3DByDirName(base,Dirname):
    fileList=os.listdir(os.path.join(base,Dirname))
    fileList=[i for i in fileList if i[-4:]=='.jpg']
    fileNumber=(len(fileList))
    chest3d=np.ones((fileNumber,512,512))

    for i in range(fileNumber):
            print('IM'+str(i+1).zfill(4)+'.jpg')
            pfile=cv2.imread(os.path.join(base,Dirname,'IM'+str(i+1).zfill(4)+'.jpg'),0)
            #print(pfile.shape)



            chest3d[i,:,:]=cv2.imread(os.path.join(base,Dirname,'IM'+str(i+1).zfill(4)+'.jpg'),cv2.IMREAD_GRAYSCALE)

    return chest3d
    #prinrt(fileList)



temp=gen3DByDirName(baseDir,'P001')
print(temp.shape)
cv2.imshow('temp',temp[0])
cv2.waitKey(0)
plt.imshow(temp[0])
plt.show()