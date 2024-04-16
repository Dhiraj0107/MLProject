import os
import sys
from dataclasses import dataclass

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras import regularizers

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            ### No of classes
            num_labels=1

            from sklearn.model_selection import train_test_split
            
            X_train = train_array.iloc[:, :-1]
            y_train = train_array.iloc[:,-1]
            X_test = test_array.iloc[:, :-1]
            y_test = test_array.iloc[:,-1]
            

            model=Sequential()
            ###first layer
            model.add(Dense(20,kernel_regularizer=regularizers.l2(0.01),input_shape=(28,)))
            model.add(Activation('leaky_relu'))
            model.add(Dropout(0.3))
            ###second layer
            model.add(Dense(10,kernel_regularizer=regularizers.l2(0.01)))
            model.add(Activation('leaky_relu'))
            model.add(Dropout(0.3))
            ###third layer
            model.add(Dense(3,kernel_regularizer=regularizers.l2(0.01)))
            model.add(Activation('leaky_relu'))
            model.add(Dropout(0.3))

            ###final layer
            model.add(Dense(num_labels))
            model.add(Activation('sigmoid'))

            model.summary()

            model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

            num_epochs = 4
            num_batch_size = 32

            y_train.head()

            model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), verbose=1)

            logging.info("Model training completed")


            test_accuracy=model.evaluate(X_test,y_test,verbose=0)

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=model
            # )

            model.save("artifacts/model.keras")

            return test_accuracy
                
            


        except Exception as e:
            raise CustomException(e,sys)