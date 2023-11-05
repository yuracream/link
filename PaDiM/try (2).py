import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec

import time
import math

os.chdir(os.path.dirname(os.path.abspath(__file__)))
start = time.time()
print("\nSTART\n")

# result_dict = {}
#
# fullpath = ["a", "b" ,"c"]
# mask = [3,65,2]
#
# for i in range(len(fullpath)):
#     result_dict[fullpath[i]] = mask[i]
#
# print(result_dict)
# sorted_results = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=False))
# print(sorted_results)
# with open('sorted_results.txt', 'w') as f:
#     for key, value in sorted_results.items():
#         f.write("{}: {}\n".format(key, value))

# # ダミーデータを生成（次元を指定）
B, C, H, W = 8, 550, 56, 56
B, C, H, W = 8, 50, 5, 5
# # B, C, H, W = 3, 2, 2, 2

import numpy as np

B, C, H, W = 3, 2, 2, 2
# B, C, H, W = 2, 2, 1, 1
dist_list = []
train_outputs = [np.random.rand(C, H * W), np.random.rand(C, C, H * W)]
embedding_vectors = np.random.rand(B, C, H * W)

x =[sample[:, i] for sample in embedding_vectors]
print(embedding_vectors)
print(x)
mean = train_outputs[0]
diff = x - mean
# 共分散行列の逆行列を計算（for文で逆行列を計算する部分）
cov_inv_list = [np.linalg.inv(train_outputs[1][:, :, i]) for i in range(H * W)]

for i in range(H * W):
    mean = train_outputs[0][:, i]
    conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
    # dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
    # dist = [mahalanobis_distance(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
    dist_list.append(dist)

print(dist_list)

# # Mahalanobis距離を計算する関数をベクトル化
# # def mahalanobis_vectorized(sample, mean, cov_inv):
# def mahalanobis_vectorized(sample, mean):
#     diff = sample - mean[:, np.newaxis]  # Broadcastingを使用して差を計算
#     # dist = np.sum(diff * np.dot(cov_inv, diff), axis=0)  # ベクトル化されたマハラノビス距離の計算
#     return diff
#     # return dist
#
# # Mahalanobis距離の計算をベクトル化
# # sample = embedding_vectors.reshape(B, C, -1)
# sample = embedding_vectors.reshape(-1, B, C)
# mean_expanded = train_outputs[0].reshape(-1, C)  # Broadcastingを使用してmeanを拡張
# cov_inv_expanded = np.linalg.inv(train_outputs[1])  # 共分散行列の逆行列を計算
#
#
# print(sample)
# print(mean_expanded)
# print(sample.shape)
# print(mean_expanded.shape)
#
# # dist_list = mahalanobis_vectorized(sample, mean_expanded, cov_inv_expanded)
# dist_list = mahalanobis_vectorized(sample, mean_expanded)
# print(dist_list.shape)
# # dist_listの形状を再変形
# dist_list = dist_list.reshape(H, W, B).transpose(2, 0, 1)  # (B, H, W)の形状に変換
#
# print(dist_list)
# print(dist_list.shape)


# train_outputs = [[0], [0]]
# train_outputs[0] = np.random.rand(C, H * W)  # 平均データ
# train_outputs[1] = np.random.rand(C, C, H * W)  # 共分散行列の逆行列
#
# # print(train_outputs[0].shape)#(50, 300)
# # print(train_outputs[1].shape)#(50, 50, 300)
# print("0: ", train_outputs[0])#(50, 300)
# print("1: ", train_outputs[1])#(50, 50, 300)
#
# # ダミーデータを生成（ランダムなembedding_vectors）
# embedding_vectors = np.random.rand(B, C, H * W)  # embedding_vectors（ランダムなデータ）
# print("em", embedding_vectors)#(8, 50, 300)
# # print(embedding_vectors.shape)#(8, 50, 300)
# # print(embedding_vectors[:, 0].shape)#(8, 300)
#
# # # ダミーデータを使って処理を実行
# dist_list = []
# for i in range(H * W):
#     # if i % int(H*W/10) == 0:
#     #     print("{}%".format(int(10*(i/int(H*W/10)))))
#     mean = train_outputs[0][:, i] #(50,)
#     conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
#     # nanをカウント
#     nan_count = np.isnan(conv_inv).sum()
#     # print(conv_inv.shape)
#     # print(type(conv_inv))
#     print("nanの数:", nan_count)
#     distances = []
#     dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
#     # dist = [mahalanobis_distance(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
#     dist_list.append(dist)
# print(dist_list)
# print(len(dist_list))

# # ダミーデータを使って処理を実行
# dist_list = []
#
# mean = train_outputs[0][:, 0] #(50,)
# print(mean)
# conv_inv = np.linalg.inv(train_outputs[1][:, :, 0])
# print(conv_inv.shape)

# A = np.array([[1, 2], [5, 6], [2, 6]])  # AをNumPyの配列に変換
# B = np.array([3, 1])  # BをNumPyの配列に変換
# print(A.shape)
# print(B.shape)
# print(A - B)
# dist = [sample[:, 0] for sample in embedding_vectors]
# print(len(dist))
# dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]

# # embedding_vectors = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
# embedding_vectors = np.random.rand(2, 3, 4)
# print(embedding_vectors)
# dist = [sample[:, 0] for sample in embedding_vectors]
# print(dist)

# # ダミーデータを生成（次元を指定）
# B, C, H, W = 8, 550, 56, 56
# B, C, H, W = 8, 50, 5, 5
# train_outputs = [[0], [0]]
# train_outputs[0] = np.random.rand(C, H * W)  # 平均データ
# train_outputs[1] = np.random.rand(C, C, H * W)  # 共分散行列の逆行列
#
# print(train_outputs[0].shape)#(50, 300)
# print(train_outputs[1].shape)#(50, 50, 300)
#
# # ダミーデータを生成（ランダムなembedding_vectors）
# embedding_vectors = np.random.rand(B, C, H * W)  # embedding_vectors（ランダムなデータ）
# print(embedding_vectors.shape)#(8, 50, 300)
# print(embedding_vectors[:, 0].shape)#(8, 300)
#
end = time.time()
execution = end - start
print("\nExecution: {:.2f}sec".format(execution))
print("\a")
print("\nEND\n")
