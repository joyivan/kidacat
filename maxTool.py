import os
import numpy as np
import pydicom
from mayavi import mlab
#import SimpleITK as sitk


import chestRotate
import matplotlib
import matplotlib.pyplot as plt
import cv2
#ll
#baseDir='/media/joyivan/0a1457c2-893b-40d4-bd3f-a5316e4b4854/CT_MD/COVID-19_Cases/'
#baseDir='/Users/lizirong/Downloads/CT_MD/COVID-19_Cases'
#sampeDir=os.listdir(baseDir)
#print(sampeDir)

def getmaxcomponent(mask_array, num_limit=10):
    # sitk方法获取连通域
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    output_ex = cca.Execute(_input)
    labeled_img = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    max_label = 0
    max_num = 0
    # 不必遍历全部连通域，一般在前面就有对应全身mask的label，减少计算时间
    for i in range(1, num_limit):
        if np.sum(labeled_img == i) < 1e6:		# 全身mask的体素数量必然很大，小于设定值的不考虑
            continue
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    maxcomponent = np.array((labeled_img == max_label)).astype(np.uint8)
    print(str(max_label) + ' num:' + str(max_num))  	# 看第几个是最大的
    return maxcomponent


def get_body(CT_nii_array):
    """
    卡CT阈值获取身体（理想情况下够用了，不过多数情况下会包括到机床部分）
    """
    # 阈值二值化，获得最大的3d的连通域
    print(CT_nii_array.shape)
    CT_array = np.copy(CT_nii_array)
    threshold_all = 200  # 卡的阈值，卡出整个身体以及机床部分
    CT_array[CT_array >= threshold_all] = 1
    CT_array[CT_array <= threshold_all] = 0
    body_mask1 = getmaxcomponent(CT_array, 10)
    return body_mask1.astype(np.uint8)



from PIL import Image
def cutBed(file):
    import numpy as np

    label_np = np.zeros_like(file).astype(np.uint8)
    label_np[file > 0] = 1

    from skimage.morphology import label
    from collections import OrderedDict

    region_volume = OrderedDict()
    label_map, numregions = label(label_np == 1, return_num=True)
    region_volume['num_region'] = numregions
    total_volume = 0
    print("region num :", numregions)
    max_region = 0
    max_region_flag = 0
    for l in range(1, numregions + 1):
        region_volume[l] = np.sum(label_map == l)  # * volume_per_volume
        if region_volume[l] > max_region:
            max_region = region_volume[l]
            max_region_flag = l
        total_volume += region_volume[l]
        print("region {0} volume is {1}".format(l, region_volume[l]))
    post_label_np = label_np.copy()
    post_label_np[label_map != max_region_flag] = 0
    post_label_np[label_map == max_region_flag] = 1

    print("total region volume is :", total_volume)

    import skimage

    kernel = skimage.morphology.disk(5)
    img_dialtion = skimage.morphology.dilation(post_label_np, kernel)
    return img_dialtion*np.array(file)
def segSlice(file):
    #input is jpg
    from lungmask import LMInferer
    import SimpleITK as sitk
    import cv2
    input_image=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    input_image=np.expand_dims(input_image,axis=2)
    print('non zero counts:',np.count_nonzero(input_image))

    print('after extend dimension, the image shape:')
    print(input_image.shape,type(input_image))
    input_image2=sitk.GetImageFromArray(input_image)
    #input_image2=sitk.ReadImage(file)
    print('after transfer to sitk dataType, the image shape:')
    print(input_image2.GetSize())
    input_image2=input_image2/255
    inferer = LMInferer(force_cpu=False)

    #input_image = sitk.ReadImage(INPUT)
    segmentation = inferer.apply(input_image2)
    print('non zero counts:', np.count_nonzero(segmentation))
    #segmentation[np.nonzero(segmentation)]=255
    plt.imshow(np.squeeze(segmentation), cmap='gray')
    plt.show()
if __name__=='__main__':
    for i in range(50):
        segSlice('/home/joyivan/Downloads/data/CT_MD/COVID-19_Cases/P001/IM'+str(i+1).zfill(4)+'.jpg')

