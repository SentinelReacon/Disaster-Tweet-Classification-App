import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Annotated, Tuple
import logging
import re
import yake


def split_data(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "xtrain"],
    Annotated[pd.DataFrame, "xtest"],
    Annotated[pd.DataFrame, "ytrain"],
    Annotated[pd.DataFrame, "ytest"]
    
]:
    
    try:
        
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True)
        logging.info("Finished splitting the data")
        
        return xtrain, xtest, ytrain, ytest
    
    except Exception as e:
        logging.error(f"Error in splitting the data {e}")
        raise e


class Preprocess():
    
    def __init__(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        
    def remove_punctuations(column_name):
        new_text = []
        for tweet in column_name:
            tweet = re.sub(r'https?:\/\/.\S+', "", tweet)
            tweet = re.sub(r'#', '', tweet)
            tweet = re.sub(r'@', '', tweet)
            tweet = re.sub(r'[^\w\s]', '', tweet)
            tweet = re.sub(r'^RT[\s]+', '', tweet)
            tweet = str(tweet).replace('[','').replace(']','').replace(';','').replace(':','')
            new_text.append(tweet)
        return new_text
    
    def keyword_extract(column_name):
        keys = []
        index = []
        for i in column_name:
            kw = yake.KeywordExtractor(lan='en', n=1)
            keywords = kw.extract_keywords(i)
            if len(keywords) == 0:
                index.append(i)
                continue
            else:
                best = keywords[-1][0]
                keys.append(best)
        return keys, index
    
    """
    Implement the function which converts the text into tensors and return the tensors.
    """
        
        
