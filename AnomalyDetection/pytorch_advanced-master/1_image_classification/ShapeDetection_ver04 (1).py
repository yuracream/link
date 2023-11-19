import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
from torchvision import models

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_dir = "./data/"

# パッケージのimport
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# 入力画像の前処理をするクラス
# 訓練時と推論時で処理が異なる


class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。


    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),  # データオーギュメンテーション
                transforms.RandomHorizontalFlip(),  # データオーギュメンテーション
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)

# 訓練時の画像前処理の動作を確認
# 実行するたびに処理結果の画像が変わる

# 1. 画像読み込み
# print(os.getcwd("./"))
image_file_path = './data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)   # [高さ][幅][色RGB]

# 2. 元の画像の表示
# plt.imshow(img)
# plt.show()

# 3. 画像の前処理と処理済み画像の表示
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = ImageTransform(size, mean, std)
img_transformed = transform(img, phase="train")  # torch.Size([3, 224, 224])

# (色、高さ、幅)を (高さ、幅、色)に変換し、0-1に値を制限して表示
img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
# plt.imshow(img_transformed)
# plt.show()


def make_datapath_list(phase="train"):
    """
    データのパスを格納したリストを作成する。

    Parameters
    ----------
    phase : 'train' or 'val'
        訓練データか検証データかを指定する

    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    print(target_path)

    path_list = []  # ここに格納する

    # globを利用してサブディレクトリまでファイルパスを取得する
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


# 実行
train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")
print(train_list)

# class HymenopteraDataset(data.Dataset):
#     """
#     アリとハチの画像のDatasetクラス。PyTorchのDatasetクラスを継承。
#
#     Attributes
#     ----------
#     file_list : リスト
#         画像のパスを格納したリスト
#     transform : object
#         前処理クラスのインスタンス
#     phase : 'train' or 'test'
#         学習か訓練かを設定する。
#     """
#
#     def __init__(self, file_list, transform=None, phase='train'):
#         self.file_list = file_list  # ファイルパスのリスト
#         self.transform = transform  # 前処理クラスのインスタンス
#         self.phase = phase  # train or valの指定
#
#     def __len__(self):
#         '''画像の枚数を返す'''
#         return len(self.file_list)
#
#     def __getitem__(self, index):
#         '''
#         前処理をした画像のTensor形式のデータとラベルを取得
#         '''
#
#         # index番目の画像をロード
#         img_path = self.file_list[index]
#         img = Image.open(img_path)  # [高さ][幅][色RGB]
#
#         # 画像の前処理を実施
#         img_transformed = self.transform(
#             img, self.phase)  # torch.Size([3, 224, 224])
#
#         # 画像のラベルをファイル名から抜き出す
#         if self.phase == "train":
#             label = img_path[30:34]
#         elif self.phase == "val":
#             label = img_path[28:32]
#
#         # ラベルを数値に変更する
#         if label == "ants":
#             label = 0
#         elif label == "bees":
#             label = 1
#
#         return img_transformed, label
#
#
# # 実行
# train_dataset = HymenopteraDataset(
#     file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
#
# val_dataset = HymenopteraDataset(
#     file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
#
# # 動作確認
# index = 0
# # print(train_dataset.__getitem__(index)[0].size())
# print(train_dataset.__getitem__(index)[1])
