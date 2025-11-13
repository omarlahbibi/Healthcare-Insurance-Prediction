import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        
        try:
            logging.info("Split train test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            
            y = np.concatenate((y_train, y_test))
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)} # for imbalanced dataset

            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    random_state=42,
                    class_weight=class_weight_dict,
                    n_jobs=-1
                    ),
                "XGBoost": XGBClassifier(
                    n_estimators=1000, learning_rate=0.05, max_depth=8, subsample=0.8,
                    colsample_bytree=0.2, random_state=42,
                    scale_pos_weight=class_weight_dict[0] / class_weight_dict[1],
                    eval_metric="auc", tree_method="hist"
                    ),
                "CatBoost": CatBoostClassifier(
                    iterations=1000, depth=8, learning_rate=0.05, random_state=42,
                    auto_class_weights='Balanced', eval_metric='AUC', verbose=False)
                    }
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score<0.7:
                raise CustomException("No best model found", sys)
            
            logging.info(f"Best model found on training/testing datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, predicted)

            return roc_auc

        except Exception as e:
            raise CustomException(e, sys)