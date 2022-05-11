import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set GPU
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.InteractiveSession(config=config)
tf.config.list_physical_devices('GPU')

import argparse
from model import  MyModel
from classifier import TestClassifier

# PACKING
class BentoML:
    
    def run_bentoml(self, tf_model, sk_model):
        classifier_service = TestClassifier() # classifier객체 생성
        classifier_service.pack('tf_model', tf_model)
        classifier_service.pack('sk_model', sk_model)

        saved_path = classifier_service.save()

        print("SAVED : ", saved_path)
        return saved_path


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    # argument_parser.add_argument('--path', type=str, default="~/serving/1/saved_model.pb", help="")
    args = argument_parser.parse_args()
  
    model = MyModel()
    bento_ml = BentoML()

    path = "~/serving/1/"

    tf_model = model.run_tf_model(path) # return tf_model - model.py
    sk_model = model.run_sk_model() # return sk_model - model.py
    save_dir = bento_ml.run_bentoml(tf_model, sk_model) # model을 넘겨줘서 bentoml 프로세스 실행

    print(f"(Bento) SAVED DIR : {save_dir}")