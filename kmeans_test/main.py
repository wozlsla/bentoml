import argparse

from kmeans import KmeansMain
from bentoml_process import BentoML


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    args = argument_parser.parse_args()
  
    kmeans = KmeansMain()
    bento_ml = BentoML()

    model = kmeans.run()  # kmeans 코드 실행 -> 모델 return
    save_path = bento_ml.run_bentoml(model) # model을 넘겨줘서 bentoml 프로세스 실행

    print(f"Saved path : {save_path}")