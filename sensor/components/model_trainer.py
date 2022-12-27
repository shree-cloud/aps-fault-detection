import os,sys
from sensor.entity import config_entity,artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
from typing import Optional
import numpy as np
import pandas as pd
from sensor import utils
from xgboost import XGBClassifier
from sklearn.metrics import f1_score



class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
                try:
                    logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
                    self.model_trainer_config = model_trainer_config
                    self.data_transformation_artifact = data_transformation_artifact
                except Exception as e:
                    raise SensorException(e, sys)
    
    def fine_tune(self):
        try:
            #write code for Grid Search CV
            pass
        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self,x,y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr")
            x_train, y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info(f"Training Model")
            model = self.train_model(x=x_train,y=y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train,y_pred=yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f"train score:{f1_train_score} and test score: {f1_test_score}")
            #check for overfitting and underfitting or expected score
            logging.info(f"Checking if the model is underfiting or not")
            if f1_test_score < self.model_trainer_config.excpected_score:
                raise Exception(f"Model is not good as it is unable to give \
                expected accuracy: {self.model_trainer_config.excpected_score}; actual model score:{f1_test_score}")

            logging.info(f"Checking if the model is overfitting or not")
            diff = abs(f1_train_score - f1_test_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test score diff: {diff} is more than Overfitting Threshold {self.model_trainer_config.overfitting_threshold}")
            
            #save the trained model
            logging.info(f"Saving Model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Preparing the Artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)

# model_trainer_config = config_entity.ModelTrainerConfig
# data_transformation_artifact = artifact_entity.DataTransformationArtifact

# mt = ModelTrainer(model_trainer_config,data_transformation_artifact)
# mt.initiate_model_trainer()
