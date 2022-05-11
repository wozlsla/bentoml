import os

class PathConfig:
    def __init__(self):
        self.project_path = os.getcwd()
        self.kmeans_path = f"{self.project_path}/project_dir/"


# class EnvConfig:
#     def get_gender_mapping_code(self):
#         gender_mapping_info = {
#             'male' : 0,
#             'female' : 1,
#         }
#         return gender_mapping_info
