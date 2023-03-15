import torch
import argparse
from PIL import Image
import numpy as np
import os
from calMeanStd import png2rgb




def checkShape(train_data):
    result=[]
    for X in train_data:
        print("fileName is",X )
        filename=X
        X = png2rgb(X)
        print("shape is ",X.shape)
        shape1,_,_=X.shape
        if shape1!=512:
            print("shape must 512 512 3")
            result.append(filename)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-baseDIR', type=str, default='/media/joyivan/0a1457c2-893b-40d4-bd3f-a5316e4b4854/covidxct',
                        help='gl')
    parser.add_argument('-orginDIR', type=str, default='3A_images', help='gl')
    parser.add_argument('-workDIR', type=str, default='/home/joyivan/work/kidacat', help='gl')
    opt = parser.parse_args()
    print(opt)
    str1='012'

    traindata = np.load(os.path.join(opt.workDIR,'resultTrain.npy'), allow_pickle=True).item()
    os.chdir(os.path.join(opt.baseDIR, opt.orginDIR))
    trainError={}
    for i in str1:
        trainError[i]=checkShape(traindata[i])

    validdata = np.load(os.path.join(opt.workDIR,'resultValid.npy'), allow_pickle=True).item()
    os.chdir(os.path.join(opt.baseDIR, opt.orginDIR))
    validError={}
    for i in str1:
        validError[i]=checkShape(validdata[i])

    testdata = np.load(os.path.join(opt.workDIR,'resultTest.npy'), allow_pickle=True).item()
    os.chdir(os.path.join(opt.baseDIR, opt.orginDIR))
    testError={}
    for i in str1:
        testError[i]=checkShape(testdata[i])



    np.save(os.path.join(opt.workDIR,'errorShape.npy'),{'trainError':trainError,'validError':validError,'testError':testError})
   # print(get_mean_and_std(traindata['0']))




