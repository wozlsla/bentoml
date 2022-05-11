import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn import svm
from sklearn import datasets
import tensorflow as tf

# Option 1: custom model with specific method call
class MyModel:
    def __init__(self):
        pass

    def run_tf_model(self, export_dir):
        tf_model = tf.keras.models.load_model(export_dir)

        # tf_model = tf.saved_model.load(export_dir, tags=None, options=None)
        # '_UserObject' object has no attribute 'predict'
        # saved_model_cli show --dir ~/serving/1 --tag_set serve --signature_def serving_default

        return tf_model

    def run_sk_model(self):
        # Load training data
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # Model Training
        sk_model = svm.SVC(gamma='scale', random_state=42)
        sk_model.fit(X, y)

        return sk_model


