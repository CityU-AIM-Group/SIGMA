import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
class SPClustering(nn.Module):

    def __init__(self,d=256, k =10):
        super(SPClustering, self).__init__()
        self.d = d
        self.k = k

    def myKNN(self, S, k, sigma=1.0):
        N = len(S)
        A = np.zeros((N, N))
        for i in range(N):
            dist_with_index = zip(S[i], range(N))
            dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
            neighbours_id = [dist_with_index[m][1] for m in range(k + 1)]  # xi's k nearest neighbours

            for j in neighbours_id:  # xj is xi's neighbour
                A[i][j] = np.exp(-S[i][j] / 2 / sigma / sigma)
                A[j][i] = A[i][j]  # mutually

        return A

    def calLaplacianMatrix(self, adjacentMatrix):

        # compute the Degree Matrix: D=sum(A)
        degreeMatrix = np.sum(adjacentMatrix, axis=1)
        # print degreeMatrix
        # compute the Laplacian Matrix: L=D-A
        laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

        # normailze
        # D^(-1/2) L D^(-1/2)
        sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
        return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)


    def euclidDistance(self, x1, x2, sqrt_flag=False):
        res = np.sum((x1-x2)**2)
        if sqrt_flag:
            res = np.sqrt(res)
        return res

    def calEuclidDistanceMatrix(self, X):
        X = np.array(X)
        S = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                S[i][j] = 1.0 * self.euclidDistance(X[i], X[j])
                S[j][i] = S[i][j]
        return S

    def forward(self, nodes, labels):



        Similarity = self.calEuclidDistanceMatrix(nodes)
        Adjacent = self.myKNN(Similarity, k=self.k)
        Laplacian = self.calLaplacianMatrix(Adjacent)
        x, V = np.linalg.eig(Laplacian)
        x = zip(x, range(len(x)))
        x = sorted(x, key=lambda x: x[0])
        H = np.vstack([V[:, i] for (v, i) in x]).T
        sp_kmeans = KMeans(n_clusters=2).fit(H).cluster_centers_

