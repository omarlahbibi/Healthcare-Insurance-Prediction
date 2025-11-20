import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("TRAINING PIPELINE STARTED")

            logging.info("Starting data ingestion...")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train: {train_path}, Test: {test_path}")

            logging.info("Starting data transformation...")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path, test_path
            )
            logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")

            logging.info("Starting model training...")
            model_trainer = ModelTrainer()
            roc_auc = model_trainer.initiate_model_trainer(train_arr=train_arr, test_arr=test_arr)
            logging.info(f"Model training completed. ROC-AUC Score: {roc_auc}")

            logging.info("TRAINING PIPELINE FINISHED SUCCESSFULLY")

        except Exception as e:
            raise CustomException(e, sys)