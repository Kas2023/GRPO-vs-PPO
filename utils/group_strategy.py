import torch as th
import numpy as np
from sklearn.cluster import KMeans
    
class KMeansGroupingCPU:
    def __init__(self, n_clusters=5):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
    
    def __call__(self, features):
        if isinstance(features, th.Tensor):
            features = features.cpu().numpy()
        
        return self.kmeans.fit_predict(features)