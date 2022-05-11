from preprocess import KmeansPreprocess
from config import PathConfig
from dataio import DataIOSteam
from model import KmeansModeling


class KmeansMain(KmeansPreprocess, PathConfig, KmeansModeling, DataIOSteam):
    def __init__(self):
        KmeansPreprocess.__init__(self)
        PathConfig.__init__(self)
        KmeansModeling.__init__(self)
        DataIOSteam.__init__(self)

    def run(self):
        data = self._get_data(self.kmeans_path) # img_list_png
        data = self.run_preprocessing(data) # .npy
        # X = self._get_X_y(data) # return sample_data_2d

        kmeans_model = self.run_sklearn_modeling(data)
        return kmeans_model


