import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 id: int,
                 Gender: str,
                 Age: int,
                 Driving_Licence: int,
                 Region_Code: float,
                 Previously_Insured: int,
                 Vehicle_Age: str,
                 Vehicle_Damage: str,
                 Annual_Premium: float,
                 Policy_Sales_Channel: float,
                 Vintage: int):
        
        self.id = id
        self.Gender = Gender
        self.Age = Age
        self.Driving_Licence = Driving_Licence
        self.Region_Code = Region_Code
        self.Previously_Insured = Previously_Insured
        self.Vehicle_Age = Vehicle_Age
        self.Vehicle_Damage = Vehicle_Damage
        self.Annual_Premium = Annual_Premium
        self.Policy_Sales_Channel = Policy_Sales_Channel
        self.Vintage = Vintage

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "id": [self.id],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_Licence": [self.Driving_Licence],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)