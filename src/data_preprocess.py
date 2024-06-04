import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Annotated, Tuple
import logging
import re
import yake
from load_data import ingest_data
import torch


def split_data(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "xtrain"],
    Annotated[pd.DataFrame, "xtest"],
    Annotated[pd.DataFrame, "ytrain"],
    Annotated[pd.DataFrame, "ytest"]
    
]:
    
    try:
        X.drop(["location"], axis=1)
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
        
    def remove_punctuations(self, df: pd.DataFrame, column_name: str) -> Annotated[pd.DataFrame, "Dataframe with cleaned tweets"]:
        new_text = []
        for tweet in df[column_name]:
            tweet = re.sub(r'https?:\/\/.\S+', "", tweet)
            tweet = re.sub(r'#', '', tweet)
            tweet = re.sub(r'@', '', tweet)
            tweet = re.sub(r'[^\w\s]', '', tweet)
            tweet = re.sub(r'^RT[\s]+', '', tweet)
            tweet = str(tweet).replace('[','').replace(']','').replace(';','').replace(':','')
            new_text.append(tweet)
        df[column_name] = new_text
        return df
    
    def preprocess_nan(self, df: pd.DataFrame) -> Annotated[pd.DataFrame, "filled nan values"]:
        
        df_nan = df[df['keyword'].isnull()==True]
        df.dropna(subset=["keyword"], inplace=True)
        keys = []
        index = []
        for i in df_nan['text']:
            kw = yake.KeywordExtractor(lan='en', n=1)
            keywords = kw.extract_keywords(i)
            if len(keywords) == 0:
                index.append(i)
                continue
            else:
                best = keywords[-1][0]
                keys.append(best)
                
        df_nan["keyword"] = keys
        df_new = pd.concat([df, df_nan])
                
        return df_new
    
    """
    Implement the function which converts the text into tensors and return the tensors.
    """
    
    def convert(self, X: pd.DataFrame, y: pd.DataFrame)-> Tuple[
        Annotated[pd.DataFrame, "X tensor"],
        Annotated[pd.DataFrame, "Y tensor"]
    ]:
        
        """
        first we need to convert the text into their embeddings, then converting everything into tensor.
        """
        X_tensor = torch.from_numpy(X.values)
        y_tensor = torch.from_numpy(y.values)
        
        return X_tensor, y_tensor        
        
        
     
# experimenting stuff   
X, y = ingest_data("/home/amogh/College/Disaster Tweet Web App/data/train_tweets.csv")
xtrain, xtest, ytrain, ytest = split_data(X, y)
pre = Preprocess(xtrain, xtest, ytrain, ytest)
new = pre.remove_punctuations(xtrain, "text")
new1 = pre.preprocess_nan(new)
xten, yten = pre.convert(new1, ytrain)
print(xten)


