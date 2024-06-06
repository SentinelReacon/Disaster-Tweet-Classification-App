import logging 
import pandas as pd
from zenml import step
from torch.utils.data import DataLoader
from src.data_preprocess import split_data, Preprocess, TweetDataset, collate_batch

@step
def process_data(X: pd.DataFrame, y: pd.DataFrame):
    
    try:
        xtrain, xtest, ytrain, ytest = split_data(X, y)
        pre = Preprocess(xtrain, xtest, ytrain, ytest)
        new = pre.remove_punctuations(xtrain, "text")
        new1 = pre.preprocess_nan(new)
        text_pipe, keyword_pipe, comb_df = pre.convert(new1, ytrain)

        tw = TweetDataset(comb_df, text_pipeline=text_pipe, keyword_pipeline=keyword_pipe)
        dataloader = DataLoader(tw, collate_fn=collate_batch)
        
        logging.info("DataLoader ready to feed to the model.")    
        return dataloader
    
    except Exception as e:
        logging.error(f"Error in processing the data {e}")
        raise e