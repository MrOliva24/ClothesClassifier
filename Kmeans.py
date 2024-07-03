__authors__ = ['1668300']
__group__ = '3'

import numpy as np
import utils
from copy import deepcopy


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iters = 0
        self.K = K
        self._init_X(X)
        self._init_options(options) # DICT options
        self._init_centroids()
        self.TOLWCD = 20

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        
        X = X.astype('float')
        self.X = X.reshape(X.shape[0] * X.shape[1], 3)
        """
        X = X.astype(float)
        if X.ndim != 2:
            self.X = np.reshape(X, (X.shape[0] * X.shape[1], 3))
        else:
            self.X = X
    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf 
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_centroids(self): #bé
        """
        Initialization of centroids
        """

        self.centroids = np.zeros([self.K, 3])
        centr = np.zeros([self.K,3])
        if self.options['km_init'].lower() == 'first': 
            myList = np.empty((0, 3), dtype=self.X.dtype)  
            amp = 0
            for i in range(self.K):
                flag = False
                while not flag:
                    flag = True
                    newcentroid = self.X[i + amp]
                    if np.any(np.all(newcentroid == myList, axis=1)):
                        flag = False
                        amp += 1
                myList = np.vstack((myList, newcentroid))  
                centr[i] = newcentroid

            self.centroids = centr
            self.old_centroids = np.copy(centr)

        elif self.options['km_init'].lower() == 'random':
            indexes = []
            for i in range(self.K):
                r = np.random.randint(self.X.shape[0])
                while r in indexes:
                    r = np.random.randint(self.X.shape[0])
                centr[i] = self.X[r]
                indexes.append(r)
            self.centroids = centr
            self.old_centroids = centr #revisar
            #revisar si és obligatori fer que els centroides valguin diferent
            #(no només index diferent sino valor diferent)

        elif self.options['km_init'].lower() == 'line': #done
            self.centroids = np.zeros([self.K, 3])
            self.old_centroids = np.zeros([self.K, 3])
            for i in range(self.K):
                z = 255*(i+1)/(self.K+1)
                #print(f"z: {z} i: {i}")
                self.centroids[i] = [z,z,z]
                self.old_centroids[i] = [z,z,z] #revisar
            #cal repartir els punts de forma equidistant per la diagonal
            #del cub de colors [0,0,0] a [255,255,255]
            
        elif self.options['km_init'].lower() == 'ellipse':
            self.centroids = np.zeros([self.K, 3])
            self.old_centroids = np.zeros([self.K, 3])
            k = self.K
            pi = np.pi
            a = []
            phi = 0
            r = 80
            for i in range(k):
                x = np.round(r * np.cos(i * 2 *pi /k + phi) + 122.5, 2)
                y = np.round(r * np.sin(i * 2 *pi /k + phi) + 122.5, 2)
                z = np.round(-x-y+255+122.5,2)
                self.centroids[i] = [x,y,z]
                self.old_centroids[i] = [x,y,z]
            
            

    def get_labels(self): #FIXED
        """
        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        dist = distance(self.X, self.centroids)  
        self.labels = np.argmin(dist, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)
        for i in range(self.K):
            mask = (self.labels == i)
            points = self.X[mask]
            if len(points) > 0:
                self.centroids[i] = np.mean(points, axis=0)
            else:
                self.centroids[i] = self.old_centroids[i]

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return (self.centroids == self.old_centroids).all()

    def fit(self): #funciona
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self.num_iters = 0
        self._init_centroids()
        iters = 0
        converged = False
        while iters < self.options['max_iter'] and not converged:
            self.get_labels()
            self.get_centroids()
            iters +=1
            converged = self.converges()
        self.num_iters = iters 

    def withinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        self.get_labels()
    
        return np.sum((self.X - self.centroids[self.labels])**2) / self.X.shape[0]
    
    def externClassDistance(self):
        dist = distance(self.centroids, self.centroids)
        ECD = 0
        count = 1
        for i in range(self.K - 1):
            for j in range(i+1,self.K):
                ECD += dist[i][j]
            count += 1
        ECD /= count
        return ECD
    '''
    def externClassDistance(self):
        
        dist_matrix = np.sqrt(((self.centroids[:, np.newaxis, :] - self.centroids) ** 2).sum(axis=2))
        
        upper_tri = np.triu(dist_matrix, k=1)
        ECD = upper_tri.sum() / ((self.K * (self.K - 1) // 2))
    
        return ECD
    '''
    
    def WCDoverECD(self):
        WCD = self.withinClassDistance()
        ECD = self.externClassDistance()
        return np.sqrt(self.K) * WCD / ECD
    
    def silhouette(self):
        WCD = []
        dist = distance(self.X,self.centroids)
        for index in range(self.K):
            distclass=0
            N=0
            for i in range(self.X.shape[0]):
                N+=1
                if self.labels[i]==index:
                    dt = np.array([dist[i,index]])
                    distclass +=dist[i,index]*dt.T
            WCD.append(distclass/N)
        for i in range(self.K):
            dist[i][i]=np.inf 
        s=0
        for centroid in range(self.K):
            centroid_wcd = WCD[centroid]
            closest_centroid_index = np.where(dist[centroid]==np.min(dist[centroid]))
            closest_centroid = self.centroids[closest_centroid_index[0]][0]
            centroid_ecd = 0
            for w in range(len(self.centroids[0])):
                centroid_ecd=np.power(self.centroids[centroid][w]-closest_centroid[w],2)
            s += (centroid_ecd-centroid_wcd)/max(centroid_ecd,centroid_wcd)
        s_mean = s/self.K
        return s_mean
    


    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        self.K = 1
        self.fit()
        if(max_K<2):
            return
        if (self.options['fitting'].lower()=="wcd"):
            old_value = self.withinClassDistance()
            for K in range(2,max_K+1):
                self.K = K
                self._init_centroids()
                self.fit()
                new_value = self.withinClassDistance()
                if(new_value<old_value*0.30):
                    return
                if(new_value>old_value*0.85):
                    self.K=K-1
                    return
                old_value=new_value
        if self.options['fitting'].lower() == 'silhouette':
            old_value = 0.86
            for K in range(2,max_K+1):
                self.K = K
                self._init_centroids()
                self.fit()
                new_value = self.silhouette()[0]
                if(new_value<old_value):
                    self.K=K-1
                    return
                if(new_value>0.965 or (new_value-old_value)<0.01):
                    return
                old_value=new_value
        if self.options['fitting'].lower() == 'ecd':
            old_value = self.externClassDistance()
            for K in range(2,max_K+1):
                self.K = K
                self._init_centroids()
                self.fit()
                new_value = self.externClassDistance()
                if((new_value-old_value)/old_value<0.2):
                    print("Best K is "+str(K))
                    return
                if(K>2 and (new_value-old_value)/old_value>1):
                    self.K=K-1
                    return
                old_value=new_value
        
        

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    return np.sqrt(((X[:, np.newaxis, :] - C) ** 2).sum(axis=2))



def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    return [utils.colors[np.argmax(k)] for k in utils.get_color_prob(centroids)]
