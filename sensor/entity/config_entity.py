import os
from sensor.exception import SensorException


FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"


class TrainingPipelineConfig:

    def __init__(self):
        self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")


class DataIngestationConfig:

    def __init__(self,training_pipeline_config:TrainingPipeline):
        self.database_name="aps"
        self.collection_name="sensor"
        self.data_ingestion_dir=os.path.join(training_pipeline_config.artifcat_dir,"data_ingestion")
        self.feature_store_dir = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
        self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",Test_FILE_NAME)

    def to_dict()->dict:
        try:
            return sel
        except Exception as e:
            raise SensorException(e,sys)




class DataValidationConfig:...
class DataTransformationConfig:...
class ModelTrainerConfig:...
class ModelEvaluationConfig:...
class ModelPusherConfig:...