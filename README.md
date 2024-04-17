## Flight Cancellation Prediction


### Overview

This project aims to predict whether a flight will be canceled or not using deep learning techniques. It utilizes an imbalanced dataset, which has been addressed through the Synthetic Minority Over-sampling Technique (SMOTE) for oversampling. The model is deployed as a Docker container and can be deployed to AWS ECR and EC2 for scalable deployment.


### Features

- Predicts flight cancellation.
- Handles imbalanced dataset using SMOTE.
- Deployable as a Docker image.
- Integration with AWS ECR and EC2 for scalable deployment.


### Project Files

#### exception.py

This file contains custom exception classes that are used for raising specific errors during the execution of the application. These custom exceptions help in handling errors more effectively and provide meaningful error messages to the users.

#### logger.py

The logger.py file implements logging functionality to record all activities and events occurring during the execution of the application. It creates log files named with the date and time of their creation, helping in easy tracking and troubleshooting of issues.

#### utils.py

The utils.py file contains utility functions used for saving and retrieving important artifacts generated during the execution of the application. It handles the saving and loading of the preprocessor.pkl file, which stores the preprocessing steps applied to the data, and the model.keras file, which stores the trained deep learning model. These artifacts are crucial for maintaining reproducibility and reusability of the model.

#### setup.py

The setup.py file is used to define the project metadata and dependencies required for the application. It allows for easy installation of the application and its dependencies using package management tools like pip.

#### requirements.txt

The requirements.txt file lists all the Python packages and their versions required for running the application. It ensures that the necessary dependencies are installed before executing the application.

#### data_ingestion.py

The data_ingestion.py file handles the process of data ingestion, including reading the raw data, saving it to a CSV file, splitting it into training and testing sets, and calling the data_transformation.py and model_trainer.py files to perform exploratory data analysis (EDA) and model training, respectively. It orchestrates the overall data processing pipeline.

#### data_transformation.py

The data_transformation.py file contains functions for performing exploratory data analysis (EDA) and data transformation tasks. It applies Synthetic Minority Over-sampling Technique (SMOTE) for handling imbalanced data and saves the preprocessing steps to the preprocessor.pkl file for reuse. Additionally, it returns the transformed data back to data_ingestion.py for further processing.

#### model_trainer.py

The model_trainer.py file is responsible for creating and training the deep learning model to predict flight cancellation status. It addresses the binary classification problem of predicting whether a flight will be canceled or not. After training, it saves the trained model to the model.keras file for future use and returns the test accuracy back to data_ingestion.py, which prints it to the console.

#### Dockerfile

The Dockerfile defines the instructions for building a Docker image for the application. It specifies the base image, sets up the environment, copies the necessary files into the image, and configures the container to run the application.

#### main.yaml

The main.yaml file contains the configuration for the CI/CD pipeline using GitHub Actions. It automates the deployment process to Amazon ECR and EC2 whenever changes are made to the GitHub repository. The pipeline ensures seamless integration and deployment of the application.


### Next Steps

#### After creating the above files, created a virtual environment and moved these files to the GitHub repository from VSCode using the below commands.

##### Create Virtual environment:
- conda create -p venv python==3.9.12 -y
- conda activate venv/

##### Run the app:
- python src/components/data_ingestion.py

##### Move to GitHub Repository:
- git init
- git add README.md
- git config --global user.email "dhiraj0107@gmail.com"
- git config --global user.name "Dhiraj0107"
- git commit -m "first commit"
- git branch -M main
- git remote add origin https://github.com/Dhiraj0107/MLProject.git
- git remote -v
- git pull
- git add .
- git status
- git commit -m "Add yaml file for workflow"
- git push -u origin main

#### Next, deployed the application to Docker using the below command.
- docker build -t flight-delay .

#### Command to check the image.
- docker images

#### Created IAM user with required policies to work with ECR and EC2.

![image](https://github.com/Dhiraj0107/MLProject/assets/118677714/f25998f7-fb42-4512-ad12-e84f2f7cadb2)

![image](https://github.com/Dhiraj0107/MLProject/assets/118677714/607e4331-4ae6-4be0-bd58-4b5a3dffdee2)

#### Created private repository in Elastic Container Registry (ECR) for uploading docker image.

![image](https://github.com/Dhiraj0107/MLProject/assets/118677714/eae2694e-5a85-45fc-baa9-4360ab9a84d3)

#### Created EC2 instance to deploy the application to Ubuntu virtual server and deployed the application

![image](https://github.com/Dhiraj0107/MLProject/assets/118677714/a3b7f043-8a92-4975-9506-d6377896e4f4)

![image](https://github.com/Dhiraj0107/MLProject/assets/118677714/58c9e87d-99f6-4e71-aeb5-e193b32ab188)

#### CI/CD Pipeline

![image](https://github.com/Dhiraj0107/MLProject/assets/118677714/8c079991-e36a-451f-b7b0-8924c9637ea0)

#### Runner for running deployment jobs

![image](https://github.com/Dhiraj0107/MLProject/assets/118677714/dfe913f8-b4a7-44f9-a31c-334139c459d7)

#### Repository secrets used in main.yaml file for CI/CD automation

![image](https://github.com/Dhiraj0107/MLProject/assets/118677714/172b6ffc-a132-4524-b3e0-aa0bdd125fa0)




