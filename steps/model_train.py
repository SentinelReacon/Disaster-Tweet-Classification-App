import logging
import pandas as pd
from zenml import step
from zenml.client import Client
import mlflow
import torch

from src.train_model import model_training

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker)
def model_train(dataloader):
    
    try:
        mlflow.sklearn.autolog()
        model = model_training(dataloader=dataloader)
        
        torch.save(model.state_dict(), "/home/amogh/College/Disaster Tweet Web App/models/")
        
        return model
    
    except Exception as e:
        logging.error(f"Error in training the model {e}")
        raise e
