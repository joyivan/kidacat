import torch
import argparse
from PIL import Image
import numpy as np
import os
import threading
def png2rgb(filename):
   im=Image.open(filename)
   im=im.convert('RGB')
   return np.array(im)

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
        print('In MyThread----------------------------------------------')
    def get_result(self):
       # threading.Thread.join(self)  # 等待线程执行完毕
        #try:
            return self.result
        #except Exception:
        #    return None


def get_mean_and_std(fileList):
    print('totalFile is:',len(fileList))
    mean = np.zeros(3)
    std =np.zeros(3)

    for X in fileList:
        print("fileName is",X )
        X=png2rgb(X)
        #print("shape is ",X.shape)
        shape1,_,_=X.shape
        if shape1!=512:
            raise ValueError("shape must 512 512 3")
        for d in range(3):
            mean[d] += X[:,:, d].mean()
            std[d] += X[:, :, d].std()
    mean=mean/(len(fileList))
    std=std/(len(fileList))
    return mean, std
def getFinalMean_Std(totalData,errorShape,belonesString):
    dataSet=set()
    for i in '012':
       print(type(totalData[i]))
       dataSet=dataSet|(set(totalData[i]))
    # print('dataSet is :',dataSet)
    Result=dataSet-set(errorShape[belonesString+'Error']['0'])-set(errorShape[belonesString+'Error']['1'])-\
           set(errorShape[belonesString+'Error']['2'])

    Result=list(Result)

    mean,std=get_mean_and_std(Result)
    rr=[mean,std]
    return rr
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-baseDIR', type=str, default='/media/joyivan/0a1457c2-893b-40d4-bd3f-a5316e4b4854/covidxct',
                        help='gl')
    parser.add_argument('-orginDIR', type=str, default='3A_images', help='gl')
    parser.add_argument('-workDIR', type=str, default='/home/joyivan/work/kidacat', help='gl')
    opt = parser.parse_args()
    print(opt)
    os.chdir(os.path.join(opt.baseDIR,opt.orginDIR))
    errorShape= np.load(os.path.join(opt.workDIR,'errorShape.npy'),allow_pickle=True).item()
   #{'trainError':trainError,'validError':validError,'testError':testError}) sublist
    traindata = np.load(os.path.join(opt.workDIR, 'resultTrain.npy'), allow_pickle=True).item()
    #dict (key:0,1,2)
    validdata = np.load(os.path.join(opt.workDIR, 'resultValid.npy'), allow_pickle=True).item()
    testdata = np.load(os.path.join(opt.workDIR, 'resultTest.npy'), allow_pickle=True).item()
    train=MyThread(getFinalMean_Std,(traindata,errorShape,'train',))
    valid = MyThread(getFinalMean_Std, (validdata, errorShape, 'valid',))
    test = MyThread(getFinalMean_Std, (testdata, errorShape, 'test',))

    train.start()
    train.run()
    valid.start()
    valid.run()
    test.start()
    test.run()
    trainresult=train.get_result()
    trainresult=list(trainresult)
    trainMean=trainresult[0]
    trainStd=trainresult[1]
    print("train:",trainMean,trainStd)
    validresult = valid.get_result()
    validresult = list(validresult)
    validMean = validresult[0]
    validStd = validresult[1]
    print("valid:", validMean, validStd)
    testresult = test.get_result()
    testresult = list(testresult)
    testMean = testresult[0]
    testStd = testresult[1]
    print("test:", testMean, testStd)

    result={'trainSet':[trainMean,trainStd],'validSet':[validMean,validStd],'testSet':[testMean,testStd]}
    os.chdir(opt.workDIR)
    np.save('resultMeanStd.npy',result)









