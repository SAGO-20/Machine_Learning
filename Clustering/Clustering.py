# Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")


# Setting seed to 123
random.seed(1234)

np.random.seed(123)

# Importing the dataset
dataset = pd.read_csv('dataset', delimiter= ' ', header = None)

# Dropping the first column
df = dataset.drop(0,axis='columns')

# Converting into numpy array
X = df.values

# Silhoutte Coefficient
def silhouette_coefficient(X,labels):

    n_samples, n_features = X.shape

    a = np.zeros(n_samples)
    b = np.zeros(n_samples)

    unique_clusters = np.unique(labels)

    no_unique_clusters = len(unique_clusters)

    # Computing value of a
    for i in range(n_samples):
        cluster_label = labels[i]
        mask = (labels == cluster_label)
        size = np.sum(mask)
        if size == 1:
            a[i] = 0
        else:
            cluster_data = X[mask]
            distances = np.sqrt(np.sum((cluster_data - X[i])**2, axis = 1))
            a[i] = sum(distances)/(len(distances) - 1 )
        

    # Computing value of b
    for i in range(n_samples):
        cluster_label = labels[i]
        min_dist = np.inf
        for j in range(no_unique_clusters):
            if unique_clusters[j] != cluster_label:
                mask = (labels == unique_clusters[j])
                cluster_data = X[mask]
                distances = np.sqrt(np.sum((cluster_data - X[i])**2,axis=1))
                if len(distances) > 0:
                    d = np.mean(distances)
                    if d < min_dist:
                        min_dist = d
        b[i] = min_dist
                        

    # Computing silhoutte coefficient
    s = np.zeros(n_samples)
    for i in range(n_samples):
        if b[i] > 0 and a[i] > 0:
            s[i] = (b[i] - a[i])/max(b[i],a[i])
    return np.mean(s)


#1 
# K-Means Clustering
class K_Means:

    def __init__(self, K , max_iterations = 100):
        self.K = K
        self.max_iterations = max_iterations

    def fit_predict(self,X):
        self.X = X
        labels = []

        # Shape of data
        n_samples, n_features = self.X.shape  # n_samples = 300, n_features = 337

        # Initialise the cluster centroids
        row_indxs = random.sample(range(0,n_samples),self.K)
        self.centroids = np.array([X[indx] for indx in row_indxs], dtype = 'float')
        
        for i in range(self.max_iterations): 
            # Assign the data points to cluster
            cluster_labels = self._assign_clusters(self.X)
        
            # Optimise the cluster centroids
            old_centroids = self.centroids.copy()
            self.centroids = self._optimise_centroids(self.X, cluster_labels)

            # Check convergence
            if self._centroids_convergence(old_centroids):
                break

        # Return labels
        return cluster_labels
    

    def _assign_clusters(self,X):
        labels = np.array([])
        for row in X:
            assign = []
            for centroid in self.centroids:
                distance = self._Euclidean_distance(row,centroid)
                assign.append(distance)
            cluster_selected = np.argmin(assign)
            labels = np.append(labels,cluster_selected)   
        return labels
    
    def _optimise_centroids(self, X, labels):
        for i in range(self.K):
            self.centroids[i] = np.mean(X[labels == i], axis = 0)
        return self.centroids

    def _centroids_convergence(self,old_centroids):
        return np.array_equal(self.centroids, old_centroids)
    
    def _Euclidean_distance(self,x,y):
        return np.sqrt(np.sum((x - y)**2))

#2
# K-Means Clustering ++
class K_Means_plus_plus:

    def __init__(self, K , max_iterations = 100):
        self.K = K
        self.max_iterations = max_iterations

    def fit_predict(self,X):
        self.X = X

        # Shape of data
        n_samples, n_features = self.X.shape  

        # Initialise the centroids
        self.centroids = np.empty((self.K,n_features))

        # Initialise the cluster centroids
        self._initialize_centroids(self.X,n_samples)
        
        for i in range(self.max_iterations):
            # Assign the data points to cluster
            self.cluster_labels = self._assign_clusters(self.X)
        
            # Optimise the cluster centroids
            old_centroids = self.centroids.copy()
            self.centroids = self._optimise_centroids(self.X, self.cluster_labels)

            # Check convergence
            if self._centroids_convergence(old_centroids):
                break

        # Return labels
        return self.cluster_labels
    

    def _assign_clusters(self,X):
        labels = np.array([])
        for row in X:
            assign = []
            for centroid in self.centroids:
                distance = self._Euclidean_distance(row,centroid)
                assign.append(distance)
            cluster_selected = np.argmin(assign)
            labels = np.append(labels,cluster_selected)   
        return labels
    
    def _optimise_centroids(self, X, labels):
        for i in range(self.K):
            self.centroids[i] = np.mean(X[labels == i], axis = 0)
        return self.centroids

    def _centroids_convergence(self,old_centroids):
        return np.array_equal(self.centroids, old_centroids)
    
    def _Euclidean_distance(self,x,y):
        return np.sqrt(np.sum((x - y)**2))
    
    def _initialize_centroids(self,X,n_samples):
        # Randomly choosing first centroid
        self.centroids[0] = X[np.random.choice(n_samples)]

        # Selecting next centroid using weighted probability method
        for k in range(1,self.K):
            distance_matrix = np.empty((n_samples,k))
            for i in range(k):
                distance_matrix[:,i] = np.linalg.norm(X - self.centroids[i], axis = 1)
            nearest_centroid_distance = np.min(distance_matrix,axis = 1)
            weighted_proba = ((nearest_centroid_distance) ** 2)/ np.sum((nearest_centroid_distance) ** 2)
            self.centroids[k] = X[np.random.choice(n_samples, p = weighted_proba)]

#3

# Bisecting K Means
class B_KMeans:

    def __init__(self, s):
        self.s = s
    
    def fit(self,X):
        self.X = X
        self.clusters = [self.X]
        s_c = []

        # Dividing the data into clusters 
        while len(self.clusters) < self.s:

            # Selecting the cluster with the largest error
            max_error_cluster =  max(self.clusters, key=lambda c: np.sum((c - np.mean(c, axis=0))**2))

            # Dividing the cluster into 2 using K Means
            k_means = K_Means(2)
            cluster_labels = k_means.fit_predict(max_error_cluster)
            cluster_1 = max_error_cluster[cluster_labels == 0]
            cluster_2 = max_error_cluster[cluster_labels == 1]

            # Removiing the old cluster
            self.clusters.remove(max_error_cluster)

            # Adding the newly created clusters
            self.clusters.append(cluster_1)
            self.clusters.append(cluster_2)

            # Predicting the labels
            labels = self.predict_cluster_labels()

            # Calculate the silhoutte coefficient
            coeff = silhouette_coefficient(self.X, labels)
            s_c.append(coeff)
        return s_c


    def predict_cluster_labels(self):

        cluster_labels = []

        for i in range(len(self.X)):
            for j in range(len(self.clusters)):
                for c in self.clusters[j]:
                    if np.array_equal(self.X[i],c):
                        cluster_labels.append(j)
        
        return np.array(cluster_labels)


#4
x_axis = []
y_axis = []
for k in range(2,10):
   KMeans = K_Means(K=k)
   labels = KMeans.fit_predict(X)
   coeff = silhouette_coefficient(X,labels)
   x_axis.append(k)
   y_axis.append(coeff)

plt.plot(x_axis,y_axis, label = 'K-Means')


#5
y1_axis = []
for k in range(2,10):
    KMeans_plus = K_Means_plus_plus(K=k)
    labels = KMeans_plus.fit_predict(X)
    coeff = silhouette_coefficient(X,labels)
    y1_axis.append(coeff)

plt.plot(x_axis,y1_axis, label = 'K++ Means')


#6
b_k_means = B_KMeans(s=9)
y2_axis = b_k_means.fit(X)

plt.plot(x_axis, y2_axis, label = 'Bisecting K Means')
plt.legend()
plt.show()

