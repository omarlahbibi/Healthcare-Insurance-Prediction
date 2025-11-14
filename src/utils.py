import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.exception import CustomException


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_test_proba = model.predict_proba(X_test)[:, 1]

            test_score = roc_auc_score(y_test, y_test_proba)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)