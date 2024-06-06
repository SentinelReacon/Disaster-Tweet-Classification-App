import logging
from zenml import step
import pandas as pd
from typing import Tuple, Annotated

@step
def load_data(data_path: str) -> Tuple[Annotated[pd.DataFrame, "X dataframe (without labels)"], 
                                         Annotated[pd.DataFrame, "y dataframe (labels)"]]:

    try:
        logging.info("Loading Data from the csv file")
        df = pd.read_csv(data_path)
        X = df.drop('target', axis=1)
        y = df['target']
        return X, y

    except Exception as e:
        logging.error(f"Error in loading data from the file {e}")
        raise e