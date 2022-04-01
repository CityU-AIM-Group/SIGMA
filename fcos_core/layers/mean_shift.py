''' Time test

https://github.com/thanhkaist/MeanShiftClustering/blob/master/mean-shift-pytorch-gpu.py

Num data | batch | Time        |  gpu_Mem
300        1000    3.25/3.26s      400MB
3000       1000    3.39/4.17s      727MB
30000      1000    53.47/72.25s    2583MB
30000      2000    34.89/69.89s    4641MB
30000      4000    9.42/70.17      8762MB
Hyper parameter compare to sklearn version
                    Our         | sklearn
max_iter            10               300
check_converge      No               Yes
auto_bw             No               Yes
=> Direction to go: C version https://github.com/Sydney-Informatics-Hub/GPUnoCUDA
'''
import math
import time

import numpy as np
import torch
from torch import exp, sqrt


class MeanShift_GPU():
    ''' Do meanshift clustering with GPU support'''
    def __init__(self, bandwidth=0.1, batch_size=1000, max_iter=10, eps=1e-5, check_converge=False, use_GPU=True):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.bandwidth = bandwidth
        self.eps = eps  # use for check converge
        self.cluster_eps = 1e-1  # use for check cluster
        self.check_converge = check_converge  # Check converge will take 1.5 time longer
        self.use_GPU = use_GPU  # Check converge will take 1.5 time longer

    def distance_batch(self, a, B):
        ''' Return distance between each element in a to each element in B'''
        return sqrt(((a[None, :] - B[:, None]) ** 2)).sum(2)

    def distance(self, a, b):
        return torch.sqrt(((a - b) ** 2).sum()) if self.use_GPU else np.sqrt(((a - b) ** 2).sum())

    def fit(self, data):
        with torch.no_grad():
            n = len(data)
            X = data.clone()
            # X = torch.from_numpy(np.copy(data)).cuda()
            for _ in range(self.max_iter):
                max_dis = 0
                for i in range(0, n, self.batch_size):
                    s = slice(i, min(n, i + self.batch_size))
                    if self.check_converge:
                        dis = self.distance_batch(X, X[s])
                        max_batch = torch.max(dis)
                        if max_dis < max_batch:
                            max_dis = max_batch
                        weight = dis
                        weight = self.gaussian(dis, self.bandwidth)
                    else:
                        weight = self.gaussian(self.distance_batch(X, X[s]), self.bandwidth)
                    num = (weight[:, :, None] * X).sum(dim=1)
                    X[s] = num / weight.sum(1)[:, None]

                # Check converge
                if self.check_converge:
                    if max_dis < self.eps:
                        print("Converged")
                        break
            if self.use_GPU:
                labels, centers = self.cluster_points(X)
            else:
                points = X.cpu().data.numpy()
                labels, centers = self.cluster_points(points)
            return labels, centers

    def gaussian(self, dist, bandwidth):
        return exp(-0.5 * ((dist / bandwidth)) ** 2) / (bandwidth * math.sqrt(2 * math.pi))

    def cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if (len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for j, center in enumerate(cluster_centers):
                    dist = self.distance(point, center)
                    if (dist < self.cluster_eps):
                        cluster_ids.append(j)
                if (len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        cluster_ids = torch.Tensor(cluster_ids).cuda()
        return cluster_ids, cluster_centers


