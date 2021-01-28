# -*- coding: utf-8 -*-
"""
Created with：PyCharm
@Author： Jcsim
@Date： 2021-1-28 15:44
@Project： ocr_auth_code
@File： ocrModel.py
@Blog：https://blog.csdn.net/weixin_38676276
@Description： 
@Python：
"""
# 安装torch和torchvision、opencv-python即可跑通
import io
import os
import string
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from ocr_auth_code import settings


class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        super(Model, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()

        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block + 1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


characters = string.digits + string.ascii_uppercase
width, height, n_len, n_classes = 203, 66, 6, len(characters)
n_input_length = 12
# print(characters, width, height, n_len, n_classes)
model = Model(n_classes, input_shape=(3, height, width))


# this if for store all of the image data
# this function is for read image,the input is directory name
def read_pic(filename):
    array_of_img_tensor = []
    img = to_tensor(cv2.imread(filename))
    array_of_img_tensor.append(img)
    return array_of_img_tensor


def read_pic_by_bytes(image):
    array_of_img_tensor = []
    image = Image.open(io.BytesIO(image))
    img = to_tensor(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    array_of_img_tensor.append(img)
    return array_of_img_tensor


class CaptchaDataset1(Dataset):
    def __init__(self, characters, img_tensor, input_length, label_length):
        super(CaptchaDataset1, self).__init__()
        self.characters = characters
        self.img = img_tensor
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        # self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        # random_str = ''.join([random.choice(self.characters[1:]) for j in range(self.label_length)])
        # image = to_tensor(self.generator.generate_image(random_str))
        image = self.img[index]
        input_length = torch.full(size=(1,), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1,), fill_value=self.label_length, dtype=torch.long)
        return image, input_length, target_length


def decode(sequence):
    a = ''.join([characters[x] for x in sequence])
    s = ''.join([x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j + 1]])
    if len(s) == 0:
        return ''
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


def doCheak(path):
    """

    :param path: 图片路径（ps：不能有中文）
    :return:
    """
    array_of_img_tensor = read_pic(path)
    dataset = CaptchaDataset1(characters, array_of_img_tensor, n_input_length, n_len)
    model1 = torch.load(os.path.join(settings.BASE_DIR, "backend", "6digi_ctc.pth"))
    image, input_length, label_length = dataset[0]
    output = model1(image.unsqueeze(0))
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    # print('pred:', decode(output_argmax[0]))
    return decode(output_argmax[0])


def doCheakByBytes(image):
    array_of_img_tensor = read_pic_by_bytes(image)
    dataset = CaptchaDataset1(characters, array_of_img_tensor, n_input_length, n_len)
    model1 = torch.load(os.path.join(settings.BASE_DIR, "ocr", "6digi_ctc.pth"))
    image, input_length, label_length = dataset[0]
    output = model1(image.unsqueeze(0))
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    # print('pred:', decode(output_argmax[0]))
    return decode(output_argmax[0])


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.abspath('.'))
    config_path = os.path.join(settings.BASE_DIR, "static", "img", "test", "DYJ2MS.jpg")
    # print(config_path)
    # print(root_dir)
    # # print(doCheak("DYJ2MS.jpg"))
    # print(doCheak(config_path))
    with open(os.path.join(settings.BASE_DIR, "static", "img", "test", "DYJ2MS.jpg"), "rb") as f:
        image = f.read()
        print(doCheakByBytes(image))
    # path = settings.BASE_DIR + '/static/img/' + "auth_code" + '/'
    # print(path)
    # print(os.path.exists(path))
    print(os.path.join(settings.BASE_DIR, "ocr", "6digi_ctc.pth"))
