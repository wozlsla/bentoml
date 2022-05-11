import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import ImageInput, DataframeInput
from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.frameworks.sklearn import SklearnModelArtifact

# @env(pip_packages=['imageio'])
# @env(infer_pip_packages=True)
@env(requirements_txt_file="./requirements.txt")
@artifacts([KerasModelArtifact('tf_model'), SklearnModelArtifact('sk_model')])
class TestClassifier(BentoService):
    def __init__(self):
        super().__init__()
        pass


    @api(input=DataframeInput(), batch=True)
    def iris_predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.sk_model.predict(df)


    @api(input=ImageInput(), batch=True)
    def img_predict(self, img):

        img = tf.Variable(img) # Constant ?
        img = tf.image.resize(img, [180, 180])

        print(img.shape) 
        # <bound method _EagerTensorBase.numpy of <tf.Tensor: shape=(1, 180, 180, 3), dtype=float32, numpy= ~
        # (1, 180, 180, 3)

        print(img.dtype)
        # <dtype: 'float32'>
        
        print(type(img)) 
        # <class 'tensorflow.python.framework.ops.EagerTensor'>
        
        img_np_1 = np.array(img)
        print(type(img_np_1)) # <class 'numpy.ndarray'>

        # img_np_2 = img.numpy()
        # print(type(img_np_2)) # <class 'numpy.ndarray'>

        # img = np.asarray(img) # <class 'numpy.ndarray'>

        # print(img.shape.as_list()) # 'function' object has no attribute 'shape'
        # Failed to find data adapter that can handle input: <class 'method'>, <class 'NoneType'>
        # img = np.array(img) # 'Failed to convert a NumPy array to a Tensor (Unsupported object type method).
        # Failed to find data adapter that can handle input: <class 'method'>, <class 'NoneType'>
        
        output = self.artifacts.tf_model.predict(img_np_1) 
        # '_UserObject' object has no attribute 'predict'

        return output


    # @api(input=ImageInput(), batch=True)
    # def predict(self, img):
    #     print(type(img)) # list
    #     img_arr = np.array(img, dtype=np.float32) 
    #     print(img_arr)
    #     print(img_arr.shape) # (1, 240, 180, 3)
    #     img_arr = np.reshape(1, 180, 180, 3)
    #     print(img_arr)
    #     img_arr = tf.expand_dims(img_arr, 0)
    #     print(img_arr) # shape=(1, 1, 240, 180, 3), dtype=float32
    #     output = self.artifacts.tf_model.predict(img_arr) # '_UserObject' object has no attribute 'predict'

    #     return output