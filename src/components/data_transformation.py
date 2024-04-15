import sys
from dataclasses import dataclass
import datetime

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["Airline_Name", "Origin_City", "Destination_City", "Departure_Delay_Minutes", "Arrival_Delay_Minutes", "Cancellation_Code", "Diverted_Flag", "Scheduled_Elapsed_Time_Minutes", "Actual_Elapsed_Time_Minutes", "Day", "Scheduled_Departure_Hour", "Scheduled_Departure_Minute", "Actual_Departure_Hour", "Actual_Departure_Minute", "Scheduled_Arrival_Hour", "Scheduled_Arrival_Minute", "Actual_Arrival_Hour", "Actual_Arrival_Minute", "Carrier_Delay_HH", "Carrier_Delay_MM", "Weather_Delay_HH", "Weather_Delay_MM", "NAS_Delay_HH", "NAS_Delay_MM", "Security_Delay_HH", "Security_Delay_MM", "Late_Aircraft_Delay_HH", "Late_Aircraft_Delay_MM"]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median"))

                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns)

                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,df):

        try:
            df=pd.read_csv(df)

            logging.info("Read data completed")

            logging.info("Obtaining preprocessing object")

            df['Day'] = pd.to_datetime(df['Flight_Date']).dt.day_name()
            df = df.drop(columns=['Flight_Date'])
            
            # Convert Cancelled_Flag to binary numerical format (0 for not cancelled, 1 for cancelled)
            df['Cancelled_Flag'] = df['Cancelled_Flag'].apply(lambda x: 0 if x == 'Not Cancelled' else 1)

            # Create a dictionary to store label encoders for each column
            label_encoders = {}

            # Encode categorical variables using LabelEncoder
            categorical_columns = ['Airline_Name', 'Origin_City', 'Destination_City', 'Cancellation_Code']
            for col in categorical_columns:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])
                label_encoders[col] = label_encoder

            # Convert time columns into numerical representation (e.g., seconds since midnight)
            df['Scheduled_Departure_Hour'] = df['Scheduled_Departure_Time'].apply(lambda x: int(x.split(':')[0]))
            df['Scheduled_Departure_Minute'] = df['Scheduled_Departure_Time'].apply(lambda x: int(x.split(':')[1]))

            # Drop the original 'Scheduled_Departure_Time' column
            df.drop(columns=['Scheduled_Departure_Time'], inplace=True)

            # Repeat the above process for other time columns

            
            df['Actual_Departure_Hour'] = df['Actual_Departure_Time'].apply(lambda x: int(x.split(':')[0]))
            df['Actual_Departure_Minute'] = df['Actual_Departure_Time'].apply(lambda x: int(x.split(':')[1]))

            df.drop(columns=['Actual_Departure_Time'], inplace=True)


            df['Scheduled_Arrival_Hour'] = df['Scheduled_Arrival_Time'].apply(lambda x: int(x.split(':')[0]))
            df['Scheduled_Arrival_Minute'] = df['Scheduled_Arrival_Time'].apply(lambda x: int(x.split(':')[1]))

            df.drop(columns=['Scheduled_Arrival_Time'], inplace=True)



            df['Actual_Arrival_Hour'] = df['Actual_Arrival_Time'].apply(lambda x: int(x.split(':')[0]))
            df['Actual_Arrival_Minute'] = df['Actual_Arrival_Time'].apply(lambda x: int(x.split(':')[1]))

            df.drop(columns=['Actual_Arrival_Time'], inplace=True)


            df['Carrier_Delay_HH'] = df['Carrier_Delay_HH_MM'].apply(lambda x: int(x.split(':')[0]))
            df['Carrier_Delay_MM'] = df['Carrier_Delay_HH_MM'].apply(lambda x: int(x.split(':')[1]))
            #df['Carrier_Delay_SS'] = df['Carrier_Delay_HH_MM'].apply(lambda x: int(x.split(':')[2]))

            df.drop(columns=['Carrier_Delay_HH_MM'], inplace=True)


            df['Weather_Delay_HH'] = df['Weather_Delay_HH_MM'].apply(lambda x: int(x.split(':')[0]))
            df['Weather_Delay_MM'] = df['Weather_Delay_HH_MM'].apply(lambda x: int(x.split(':')[1]))

            df.drop(columns=['Weather_Delay_HH_MM'], inplace=True)


            df['NAS_Delay_HH'] = df['NAS_Delay_HH_MM'].apply(lambda x: int(x.split(':')[0]))
            df['NAS_Delay_MM'] = df['NAS_Delay_HH_MM'].apply(lambda x: int(x.split(':')[1]))

            df.drop(columns=['NAS_Delay_HH_MM'], inplace=True)


            df['Security_Delay_HH'] = df['Security_Delay_HH_MM'].apply(lambda x: int(x.split(':')[0]))
            df['Security_Delay_MM'] = df['Security_Delay_HH_MM'].apply(lambda x: int(x.split(':')[1]))

            df.drop(columns=['Security_Delay_HH_MM'], inplace=True)


            df['Late_Aircraft_Delay_HH'] = df['Late_Aircraft_Delay_HH_MM'].apply(lambda x: int(x.split(':')[0]))
            df['Late_Aircraft_Delay_MM'] = df['Late_Aircraft_Delay_HH_MM'].apply(lambda x: int(x.split(':')[1]))

            df.drop(columns=['Late_Aircraft_Delay_HH_MM'], inplace=True)

            df['Diverted_Flag'] = df['Diverted_Flag'].apply(lambda x: 0 if x == 'Not Diverted' else 1)

            # Define a mapping dictionary for days of the week
            day_mapping = {
                'Monday': 1,
                'Tuesday': 2,
                'Wednesday': 3,
                'Thursday': 4,
                'Friday': 5,
                'Saturday': 6,
                'Sunday': 7
            }

            # Map the days of the week to numerical values
            df['Day'] = df['Day'].map(day_mapping)

            #df.head()

            X = df.drop(columns=['Cancelled_Flag'],axis=1)

            #X.head()

            y = df['Cancelled_Flag']

            #y

            # import matplotlib.pyplot as plt

            # check_classes = pd.value_counts(df['Cancelled_Flag'], sort = True)

            # check_classes.plot(kind = 'bar', rot=0)

            # plt.title("Cancelled Status Distribution")

            # plt.xlabel("Cancelled Status")

            # plt.ylabel("Frequency")

            # ## Get the Not Cancelled and Cancelled datasets

            # Not_Cancelled = df[df['Cancelled_Flag']==0]

            # Cancelled = df[df['Cancelled_Flag']==1]

            #print(Not_Cancelled.shape,Cancelled.shape)

            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Cancelled_Flag"
            numerical_columns = ["Airline_Name", "Origin_City", "Destination_City", "Departure_Delay_Minutes", "Arrival_Delay_Minutes", "Cancellation_Code", "Diverted_Flag", "Scheduled_Elapsed_Time_Minutes", "Actual_Elapsed_Time_Minutes", "Day", "Scheduled_Departure_Hour", "Scheduled_Departure_Minute", "Actual_Departure_Hour", "Actual_Departure_Minute", "Scheduled_Arrival_Hour", "Scheduled_Arrival_Minute", "Actual_Arrival_Hour", "Actual_Arrival_Minute", "Carrier_Delay_HH", "Carrier_Delay_MM", "Weather_Delay_HH", "Weather_Delay_MM", "NAS_Delay_HH", "NAS_Delay_MM", "Security_Delay_HH", "Security_Delay_MM", "Late_Aircraft_Delay_HH", "Late_Aircraft_Delay_MM"]

            input_feature_arr=X_resampled
            target_feature_df=y_resampled

            logging.info(
                f"Applying preprocessing object on the dataframe."
            )

            arr = np.c_[
                input_feature_arr, np.array(target_feature_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)