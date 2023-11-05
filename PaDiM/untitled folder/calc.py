import numpy as np
from mahalanobis_distance import mahalanobis_distance
from scipy.spatial.distance import mahalanobis

import time

# 1000次元のランダムなデータ行列を生成
sample = np.random.rand(1000, 1000)

# 平均ベクトルを計算（各次元の平均を求める）
mean_vector = np.mean(sample, axis=0)

# 共分散行列を計算（各次元間の共分散を求める）
cov_matrix = np.cov(sample, rowvar=False)  # rowvar=Falseは各列が1つの変数を表すことを指定

# 共分散行列の逆行列を計算
cov_inv = np.linalg.inv(cov_matrix)

print(sample.shape[0])
n = 100


start = time.time()
print("\nSTART\n")

for i in range(n):
    distances = []
    for i in range(sample.shape[0]):
        dist = mahalanobis(sample[i, :], mean_vector, cov_inv)
        distances.append(dist)

end = time.time()
execution = end - start
print("Execution: {:.2f} sec".format(execution))
print("\nEND\n")

start = time.time()
print("\nSTART\n")

for i in range(n):
    distances = []
    for i in range(sample.shape[0]):
        dist =  mahalanobis_distance(sample[i, :], mean_vector, cov_inv)
        distances.append(dist)

end = time.time()
execution = end - start
print("Execution: {:.2f} sec".format(execution))
print("\nEND\n")
        
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
