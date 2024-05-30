import pandas as pd
from typing import Tuple, Annotated
import numpy as np
import logging


def ingest_data(data_path: str) -> Tuple[Annotated[pd.DataFrame, "X dataframe (without labels)"], 
                                         Annotated[pd.DataFrame, "y dataframe (labels)"]]:
    
    try:
        df = pd.read_csv("/home/amogh/College/Disaster Tweet Web App/data/train_tweets.csv")
        X = df.drop('target', axis=1)
        y = df['target']
        logging.info("Data divided into X and y")
        return X, y
    
    
    except Exception as e:
        logging.error(f"Error in dividing the data into X and y {e}")
        raise e
    
    