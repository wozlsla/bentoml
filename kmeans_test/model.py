from sklearn.cluster import KMeans
import numpy as np

class KmeansModeling:
    def __init__(self):
        pass

## RUN ##
    def run_sklearn_modeling(self, data): 
        # data : sample_data_2d [numpy list]
        kmeans_model = self._get_kmeans_model()
        kmeans_model.fit(data)
        # print('kmeans_model Score : ', kmeans_model.score(X, y))
        
        print("kmeans_model.labels_", kmeans_model.labels_)
        print("kmeans_model.labels_.shape", kmeans_model.labels_.shape)
        print(np.unique(kmeans_model.labels_, return_counts=True))
        print(kmeans_model.labels_)

        return kmeans_model

## GET ##
    def _get_kmeans_model(self):
        return KMeans(n_clusters=4, random_state=42)






        

