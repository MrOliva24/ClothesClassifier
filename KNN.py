__authors__ = ['1668300']
__group__ = '3'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        # Number of images
        P = train_data.shape[0]
        M, N = train_data.shape[1], train_data.shape[2]
        D = train_data.shape[3] if len(train_data.shape) == 4 else 1
        self.train_data = train_data.reshape(P, M * N * D)


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        P, M, N  = test_data.shape[0], test_data.shape[1], test_data.shape[2] 
        D = test_data.shape[3] if len(test_data.shape) == 4 else 1
        test_data = test_data.reshape((P, N * M * D ))
        dist_matrix = cdist(test_data, self.train_data)
        nearest_neighbor_idxs = np.argsort(dist_matrix, axis=1)[:, :k]
        self.neighbors = np.array([self.labels[idxs] for idxs in nearest_neighbor_idxs], dtype=object)
       

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """

        r1 = []

        for i in range(self.neighbors.shape[0]):
            _,listaIndex,listaCompte = np.unique(self.neighbors[i],return_inverse=True,return_counts=True)

            Nice = listaCompte[listaIndex].argmax()
            r1.append(self.neighbors[i][Nice])
        return np.array(r1)
 
    
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
