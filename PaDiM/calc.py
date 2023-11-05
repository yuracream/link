import numpy as np

from scipy.spatial.distance import mahalanobis

import time
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.append('D:\python\DeepLearning\PaDiM-Anomaly-Detection-Localization-master/try')
from mahalanobis_distance import mahalanobis_distance

def calculate_mahalanobis_distance(args):
    sample, mean_vector, cov_inv = args
    dist = mahalanobis(sample, mean_vector, cov_inv)
    return dist

sample = np.random.rand(1000, 1000)
mean_vector = np.mean(sample, axis=0)
cov_matrix = np.cov(sample, rowvar=False)
cov_inv = np.linalg.inv(cov_matrix)

n = sample.shape[0]  # サンプルの数
n = 1  # サンプルの数

print("\nSTART\n")
start = time.time()

distances = []
args_list = [(sample[i, :], mean_vector, cov_inv) for i in range(n)]

with ProcessPoolExecutor() as executor:
    distances = list(executor.map(calculate_mahalanobis_distance, args_list))

end = time.time()
execution = end - start
print("Execution: {:.2f} sec".format(execution))
print("Sample Distances (Multiprocessing):", distances[:10])  # 最初の10個の距離を表示

print("\nEND\n")

# sample = np.random.rand(1000, 1000)
# mean_vector = np.mean(sample, axis=0)
# cov_matrix = np.cov(sample, rowvar=False)  # rowvar=Falseは各列が1つの変数を表すことを指定
# cov_inv = np.linalg.inv(cov_matrix)
#
# start = time.time()
# print("\nSTART\n")
#
# distances = []
# for i in range(sample.shape[0]):
#     dist = mahalanobis(sample[i, :], mean_vector, cov_inv)
#     distances.append(dist)
#
# end = time.time()
# execution = end - start
# print("Execution: {:.2f} sec".format(execution))
# print("\nEND\n")





# サンプルデータ
# sample = np.array([1.0, 2.0, 3.0], dtype=np.float64)
# print(sample.shape)
# mean = np.array([0.0, 0.0, 0.0], dtype=np.float64)
# cov_inv = np.linalg.inv(np.eye(3))  # 単位行列を使用して単純な例を示しています


# for i in range(1000):
#     # マハラノビス距離の計算
#     dist = mahalanobis_distance(sample, mean, cov_inv)
# print("マハラノビス距離:", dist)

# for i in range(1000):
#     dist = mahalanobis(sample[i], mean, cov_inv)
# print("マハラノビス距離:", dist)
