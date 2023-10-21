import numpy as np


# 旋转，axis为旋转轴，0,1,2分别代表x,y,z轴
# theta为旋转角度，单位已改为度，非弧度
# center为旋转中心，其为一维np数组[x,y,z]，默认值为图像中心点
def rotation(data, axis, theta, c=np.array([])):  # c代表旋转点
    theta = -np.pi * theta / 180
    if c.size == 0:
        c = np.array(
            [np.floor((data.shape[0] - 1) / 2), np.floor((data.shape[1] - 1) / 2), np.floor((data.shape[1] - 1) / 2)])

    s = data.shape
    mean = np.mean(data)
    # new_data = np.ones(s) * mean # 补均值
    new_data = np.zeros(s)  # 补零

    # 绕x轴旋转
    if axis == 0:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    x = i
                    y = (j - c[1]) * np.cos(theta) - (k - c[2]) * np.sin(theta) + c[1]
                    if (y < 0 or y > s[1] - 1):
                        continue
                    z = (j - c[1]) * np.sin(theta) + (k - c[2]) * np.cos(theta) + c[2]
                    if (z < 0 or z > s[2] - 1):
                        continue
                    y1 = np.floor(y).astype(int)
                    y2 = np.ceil(y).astype(int)
                    z1 = np.floor(z).astype(int)
                    z2 = np.ceil(z).astype(int)
                    dy = y - y1
                    dz = z - z1
                    new_data[i, j, k] = (data[x, y1, z1] * (1 - dy) + data[x, y2, z1] * dy) * (1 - dz) + (
                                data[x, y1, z2] * (1 - dy) + data[x, y2, z2] * dy) * dz

    # 绕y轴旋转
    elif axis == 1:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    y = j
                    x = (i - c[0]) * np.cos(theta) - (k - c[2]) * np.sin(theta) + c[0]
                    if (x < 0 or x > s[0] - 1):
                        continue
                    z = (i - c[0]) * np.sin(theta) + (k - c[2]) * np.cos(theta) + c[2]
                    if (z < 0 or z > s[2] - 1):
                        continue
                    x1 = np.floor(x).astype(int)
                    x2 = np.ceil(x).astype(int)
                    z1 = np.floor(z).astype(int)
                    z2 = np.ceil(z).astype(int)
                    dx = x - x1
                    dz = z - z1
                    new_data[i, j, k] = (data[x1, y, z1] * (1 - dx) + data[x2, y, z1] * dx) * (1 - dz) + (
                                data[x1, y, z2] * (1 - dx) + data[x2, y, z2] * dx) * dz

    # 绕z轴旋转
    else:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    z = k
                    x = (i - c[0]) * np.cos(theta) - (j - c[1]) * np.sin(theta) + c[0]
                    if (x < 0 or x > s[0] - 1):
                        continue
                    y = (i - c[0]) * np.sin(theta) + (j - c[1]) * np.cos(theta) + c[1]
                    if (y < 0 or y > s[1] - 1):
                        continue
                    x1 = np.floor(x).astype(int)
                    x2 = np.ceil(x).astype(int)
                    y1 = np.floor(y).astype(int)
                    y2 = np.ceil(y).astype(int)
                    dx = x - x1
                    dy = y - y1
                    new_data[i, j, k] = (data[x1, y1, z] * (1 - dx) + data[x2, y1, z] * dx) * (1 - dy) + (
                                data[x1, y2, z] * (1 - dx) + data[x2, y2, z] * dx) * dy

    return new_data