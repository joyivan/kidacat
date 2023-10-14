#lizirong
import numpy as np
import os
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-baseDIR', type=str, default='/media/joyivan/0a1457c2-893b-40d4-bd3f-a5316e4b4854/covidxct',
                        help='gl')
    parser.add_argument('-orginDIR', type=str, default='3A_images', help='gl')
    parser.add_argument('-workDIR', type=str, default='/home/joyivan/work', help='gl')
    opt = parser.parse_args()
    print(opt)
    os.chdir(os.path.join(opt.baseDIR, opt.orginDIR))



    traintxt = pd.read_csv(os.path.join(opt.baseDIR, 'train_COVIDx_CT-3A.txt'), sep=' ',header=None)
    validtxt = pd.read_csv(os.path.join(opt.baseDIR, 'val_COVIDx_CT-3A.txt'), sep=' ',header=None)
    testtxt = pd.read_csv(os.path.join(opt.baseDIR, 'test_COVIDx_CT-3A.txt'), sep=' ',header=None)
    trainclassFilename ={}
    validclassFilename ={}
    testclassFilename ={}
    for i in range(3):
        result=traintxt.loc[traintxt[1] == i][0].tolist()
        trainclassFilename[str(i)]=result
        #print(result)
    for i in range(3):
        result=validtxt.loc[validtxt[1] == i][0].tolist()
        validclassFilename[str(i)]=result
    for i in range(3):
        result=testtxt.loc[testtxt[1] == i][0].tolist()
        testclassFilename[str(i)]=result


   # print(len(classFilename))

    np.save(os.path.join(opt.workDIR,'kidacat','resultTrain.npy'), trainclassFilename)
    np.save(os.path.join(opt.workDIR,'kidacat','resultValid.npy'), validclassFilename)
    np.save(os.path.join(opt.workDIR,'kidacat','resultTest.npy'), testclassFilename)




