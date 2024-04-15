import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Flight_Date,
        Airline_Name: str,
        Origin_City: str,
        Destination_City: str,
        Scheduled_Departure_Time,
        Actual_Departure_Time,
        Departure_Delay_Minutes: int,
        Scheduled_Arrival_Time,
        Actual_Arrival_Time,
        Arrival_Delay_Minutes: int,
        # Cancelled_Flag: str,
        Cancellation_Code: str,
        Diverted_Flag: str
        Scheduled_Elapsed_Time_Minutes: int,
        Actual_Elapsed_Time_Minutes: int,
        Carrier_Delay_HH_MM,
        Weather_Delay_HH_MM,
        NAS_Delay_HH_MM,
        Security_Delay_HH_MM,
        Late_Aircraft_Delay_HH_MM):

        self.Flight_Date = Flight_Date

        self.Airline_Name = Airline_Name

        self.Origin_City = Origin_City

        self.Destination_City = Destination_City

        self.Scheduled_Departure_Time = Scheduled_Departure_Time

        self.Actual_Departure_Time = Actual_Departure_Time

        self.Departure_Delay_Minutes = Departure_Delay_Minutes
    
        self.Scheduled_Arrival_Time = Scheduled_Arrival_Time
    
        self.Actual_Arrival_Time = Actual_Arrival_Time
    
        self.Arrival_Delay_Minutes = Arrival_Delay_Minutes
    
        self.Cancellation_Code = Cancellation_Code
    
        self.Diverted_Flag = Diverted_Flag
    
        self.Scheduled_Elapsed_Time_Minutes = Scheduled_Elapsed_Time_Minutes
    
        self.Actual_Elapsed_Time_Minutes = Actual_Elapsed_Time_Minutes
    
        self.Carrier_Delay_HH_MM = Carrier_Delay_HH_MM
    
        self.Weather_Delay_HH_MM = Weather_Delay_HH_MM
    
        self.NAS_Delay_HH_MM = NAS_Delay_HH_MM
    
        self.Security_Delay_HH_MM = Security_Delay_HH_MM
    
        self.Late_Aircraft_Delay_HH_MM = Late_Aircraft_Delay_HH_MM

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Flight_Date": [self.Flight_Date],
                "Airline_Name": [self.Airline_Name],
                "Origin_City": [self.Origin_City],
                "Destination_City": [self.Destination_City],
                "Scheduled_Departure_Time": [self.Scheduled_Departure_Time],
                "Actual_Departure_Time": [self.Actual_Departure_Time],
                "Departure_Delay_Minutes": [self.Departure_Delay_Minutes],
                "Scheduled_Arrival_Time": [self.Scheduled_Arrival_Time],
                "Actual_Arrival_Time": [self.Actual_Arrival_Time],
                "Arrival_Delay_Minutes": [self.Arrival_Delay_Minutes],
                "Cancellation_Code": [self.Cancellation_Code],
                "Diverted_Flag": [self.Diverted_Flag],
                "Scheduled_Elapsed_Time_Minutes": [self.Scheduled_Elapsed_Time_Minutes],
                "Actual_Elapsed_Time_Minutes": [self.Actual_Elapsed_Time_Minutes],
                "Carrier_Delay_HH_MM": [self.Carrier_Delay_HH_MM],
                "Weather_Delay_HH_MM": [self.Weather_Delay_HH_MM],
                "NAS_Delay_HH_MM": [self.NAS_Delay_HH_MM],
                "Security_Delay_HH_MM": [self.Security_Delay_HH_MM],
                "Late_Aircraft_Delay_HH_MM": [self.Late_Aircraft_Delay_HH_MM],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)