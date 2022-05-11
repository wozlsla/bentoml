from email.mime import image
from typing import List
import numpy as np
import cv2
import PIL
import imageio

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import ImageInput, FileInput, StringInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
# from bentoml.service.artifacts.common import PickleArtifact, JSONArtifact

@env(pip_packages=['pillow', 'numpy', 'imageio'])
@env(requirements_txt_file="./requirements.txt")
@artifacts([SklearnModelArtifact('kmeans_model')])
class WaferSKlearnClassifier(BentoService):
    def __init__(self):
        super().__init__()
        pass

    @api(input=ImageInput(pilmode='L'), batch=False)
    def km_predict(self, image):
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array.reshape(-1, 120*120)
        cp_arr = image_array.copy()
        # print(cp_arr, cp_arr.shape)
        output = self.artifacts.kmeans_model.fit_predict(cp_arr)
        return output


    # @api(input=ImageInput(), batch=True)
    # def predict(self, image):   
    #     # print(image, type(image))
    #     # image = str(image)
    #     # image_bytes = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # TypeError: Can't convert object of type 'list' to 'str' for 'filename'
    #     image_array = np.array(image_bytes, dtype=np.float32) # Found array with dim 4. Estimator expected <= 2. : (1, 120, 120, 3)
    #     image_array = image_array.reshape(-1, 120*120)
    #     # print(image_array, image_array.shape)
    #     # arr = image_array.flatten() # Expected 2D array, got 1D array instead
    #     # arr = arr.T
    #     # print(arr) # [255. 255. 255. ... 255. 255. 255.]

    #     # image_array = [image_array.flatten()]
    #     results = self.artifacts.kmeans_model.predict(image_array)
        
    #     return results.labels_


    # @api(input=ImageInput(), batch=True)
    # def predict(self, pil_image):
    #     # pil_image = PIL.Image.open(pil_image)
    #     image_bytes = np.array(pil_image, dtype=np.float32) 
    #     image_array = image_bytes.reshape(-1, 120*120)
    #     print(image_array, type(image_array))
    #     print(image_array.shape)
    #     image_array2 = np.array(image_array.copy())
    #     print(image_array2, type(image_array))
    #     print(image_array2.shape)

    #     outputs = self.artifacts.kmeans_model.predict(image_array2)
    #     print(outputs)
    #     return outputs.labels_


    # @api(input=FileInput(), batch=True)
    # def predict(self, file_streams):
    #     img_arrays = []
    
    #     for fs in file_streams:
    #         try:
    #             im = Image.open(fs).convert(mode="L").resize((120,120))
    #             img_array = np.array(im)
    #             img_arrays.append(img_array)
    #         except PIL.UnidentifiedImageError:
    #             print(fs)

    #     inputs = np.stack(img_arrays, axis=0)
    #     outputs = self.artifacts.kmeans_model.predict(inputs)
        
    #     return outputs.labels_


    # @api(input=MultiImageInput(input_names=('imageX', 'imageY')), batch=True)
    # for i in image:
    #     image_array = np.array(image, dtype=np.float32) 
    #     image_array = image_array.reshape(-1, 120*120)
    #     # TypeError: predict() takes 2 positional arguments but 3 were given
