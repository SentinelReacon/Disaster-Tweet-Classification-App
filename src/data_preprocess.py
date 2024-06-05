import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Annotated, Tuple
import logging
import re
import yake
from load_data import ingest_data
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


def split_data(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "xtrain"],
    Annotated[pd.DataFrame, "xtest"],
    Annotated[pd.DataFrame, "ytrain"],
    Annotated[pd.DataFrame, "ytest"]
    
]:
    
    try:
        X.drop(["location", "id"], axis=1, inplace=True)
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
    
    

    def yield_tokens(self, data_iter):
        tokenizer = get_tokenizer('basic_english')
        for text in data_iter:
            yield tokenizer(text)
            
    def convert(self, X: pd.DataFrame, y: pd.DataFrame):
        
        text = list(X["text"])
        keyword = list(X["keyword"])
        
        tokenizer = get_tokenizer('basic_english')
        global text_vocab, keyword_vocab
        text_vocab = build_vocab_from_iterator(self.yield_tokens(text), specials=["<unk>", "<pad>"])
        keyword_vocab = build_vocab_from_iterator(self.yield_tokens(keyword), specials=["<unk>", "<pad>"])
        text_vocab.set_default_index(text_vocab["<unk>"])
        keyword_vocab.set_default_index(keyword_vocab["<unk>"])
        text_pipeline = lambda x: text_vocab(tokenizer(x))
        keyword_pipeline = lambda x: keyword_vocab(tokenizer(x))
        
        df_comb = pd.concat([X, y], axis=1)
        
        return text_pipeline, keyword_pipeline, df_comb
    

class TweetDataset(Dataset):
    
    def __init__(self, dataframe, keyword_pipeline, text_pipeline):
        self.dataframe = dataframe
        self.keyword_pipeline = keyword_pipeline
        self.text_pipeline = text_pipeline  
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        keyword = self.dataframe.iloc[idx]['keyword']
        target = self.dataframe.iloc[idx]['target']
        text_tensor = torch.tensor(self.text_pipeline(text), dtype=torch.int64)
        keyword_tensor = torch.tensor(self.keyword_pipeline(keyword), dtype=torch.int64)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return text_tensor, keyword_tensor, target_tensor
    
def collate_batch(batch):
    text_batch, keyword_batch, target_batch = zip(*batch)
    text_batch = pad_sequence(text_batch, padding_value=text_vocab["<pad>"])
    keyword_batch = pad_sequence(keyword_batch, padding_value=keyword_vocab["<pad>"])
    target_batch = torch.stack(target_batch)
    return text_batch, keyword_batch, target_batch
        
        
        
     
# experimenting stuff   
X, y = ingest_data("/home/amogh/College/Disaster Tweet Web App/data/train_tweets.csv")
xtrain, xtest, ytrain, ytest = split_data(X, y)
pre = Preprocess(xtrain, xtest, ytrain, ytest)
new = pre.remove_punctuations(xtrain, "text")
new1 = pre.preprocess_nan(new)
text_pipe, keyword_pipe, comb_df = pre.convert(new1, ytrain)

tw = TweetDataset(comb_df, text_pipeline=text_pipe, keyword_pipeline=keyword_pipe)
dataloader = DataLoader(tw, batch_size=16, collate_fn=collate_batch)
print(len(text_vocab), len(keyword_vocab))
"""
this pipeline is now working, start working on steps.
"""



