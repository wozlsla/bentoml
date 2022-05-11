import cv2
import numpy as np

class KmeansPreprocess:
    def __init__(self):
        pass

    def run_preprocessing(self, data):
        # data : img_list_png
        data_list = []

        for i in data:
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE) # load
            img_array = np.array(img)
            data_list.append(img_array)
            print(i, img_array.shape)  # 불러온 이미지의 차원 확인 (H*W*C)
            # print(img_array.T.shape)

        data_np = np.array(data_list) # list to np (.npy)
        data_np = data_np.reshape(-1, 120*120) # sample_data_2d

        return data_np

