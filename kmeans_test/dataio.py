import glob

class DataIOSteam:

    def _get_data(self, path):
        img_list = glob.glob(path + "*/*")
        img_list_png = [img for img in img_list if img.endswith(".png")]

        return img_list_png
    
    # def _get_X_y(self, data):
    #     sample_data_2d = data.reshape(-1, 120 * 120)
    #     return sample_data_2d