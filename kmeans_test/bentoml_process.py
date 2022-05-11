# from config import EnvConfig
from classifier import WaferSKlearnClassifier

# PACKING
class BentoML:
    
    def run_bentoml(self, kmeans_model):
        classifier_service = WaferSKlearnClassifier() # classifier객체 생성
        classifier_service.pack('kmeans_model', kmeans_model) # packing
        saved_path = classifier_service.save()
        print("SAVED : ", saved_path)

        return saved_path
