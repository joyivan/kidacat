# 定义变分自编码器VAE
from torch import nn
class Variable_AutoEncoder(nn.Module):

    def __init__(self):

        super(Variable_AutoEncoder, self).__init__()

        # 定义编码器
        self.Encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        # 定义解码器
        self.Decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

        self.fc_m = nn.Linear(64, 20)
        self.fc_sigma = nn.Linear(64, 20)

    def forward(self, input):
#  0805 test
        code = input.view(input.size(0), -1)
        code = self.Encoder(code)

        # m, sigma = code.chunk(2, dim=1)
        m = self.fc_m(code)
        sigma = self.fc_sigma(code)

        e = torch.randn_like(sigma)

        c = torch.exp(sigma) * e + m
        # c = sigma * e + m

        output = self.Decoder(c)
        output = output.view(input.size(0), 1, 28, 28)

        return output, m, sigma

import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
#from model import Auto_Encoder, Variable_AutoEncoder
import os

# 定义超参数
learning_rate = 1e-3
batch_size = 64
epochsize = 30
root = 'E:/学习/机器学习/数据集/MNIST'
sample_dir = "image5"

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 图像相关处理操作
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])   # 一定要去掉这句，不需要Normalize操作
])

# 训练集下载
mnist_train = datasets.MNIST(root=root, train=True, transform=transform, download=True)
mnist_train = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

# 测试集下载
mnist_test = datasets.MNIST(root=root, train=False, transform=transform, download=True)
mnist_test = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True)

# image,_ = iter(mnist_test).next()
# print("image.shape:",image.shape)   # torch.Size([64, 1, 28, 28])

device = torch.device('cuda')

# 定义并导入网络结构
VAE = Variable_AutoEncoder()
VAE = VAE.to(device)
# VAE.load_state_dict(torch.load('VAE.ckpt'))

criteon = nn.MSELoss()
optimizer = optim.Adam(VAE.parameters(), lr=learning_rate)

print("start train...")
for epoch in range(epochsize):

    # 训练网络
    for batchidx, (realimage, _) in enumerate(mnist_train):

        realimage = realimage.to(device)

        # 生成假图像
        fakeimage, m, sigma = VAE(realimage)

        # 计算KL损失与MSE损失
        # KLD = torch.sum(torch.exp(sigma) - (1 + sigma) + torch.pow(m, 2)) / (input.size(0)*28*28)
        # KLD = torch.sum(torch.exp(sigma) - (1 + sigma) + torch.pow(m, 2))
        # 此公式是直接根据KL Div公式化简，两个分布分别是(0-1)分布与(m,sigma)分布
        # 最后根据像素点与样本批次求平均，既realimage.size(0)*28*28
        KLD = 0.5 * torch.sum(
            torch.pow(m, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (realimage.size(0)*28*28)

        # 计算均方差损失
        # MSE = criteon(fakeimage, realimage)
        MSE = torch.sum(torch.pow(fakeimage - realimage, 2)) / (realimage.size(0)*28*28)

        # 总的损失函数
        loss = MSE + KLD

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batchidx%300 == 0:
            print("epoch:{}/{}, batchidx:{}/{}, loss:{}, MSE:{}, KLD:{}"
                  .format(epoch, epochsize, batchidx, len(mnist_train), loss, MSE, KLD))

    # 生成图像
    realimage, _ = iter(mnist_test).next()
    realimage = realimage.to(device)
    fakeimage, _, _ = VAE(realimage)

    # 真假图像何必成一张
    image = torch.cat([realimage[0:32], fakeimage[0:32]], dim=0)

    # 保存图像
    save_image(image, os.path.join(sample_dir, 'image-{}.png'.format(epoch + 1)), nrow=8, normalize=True)

    torch.save(VAE.state_dict(), 'VAE.ckpt')
