import torch
import numpy as np
import pandas as pd
import csv
import sys
from torch.utils.data import dataset
from datasets import Dataset, DatasetDict

# id2label = dict()
# with open('./id2label.txt', 'r') as f:
#     for line in f.readlines():
#         idx, label = line.split('\t')
#         idx = int(idx[4:])
#         label = label[6:]
#         id2label[idx] = label   
# f.close()

class MultiLabelDataset(pd.DataFrame):
    def __init__(self, 
                 df=None, 
                 tokenizer=None, 
                 max_length=1024, 
                 max_samples=None,
                 is_test=False
        ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        if max_samples is not None:
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        text = self.df.loc[index, "text"]
        text = text.replace('</s>','<s>')
        labels = self.df.loc[index, "labels"]
        # process caption
        model_inputs = self.tokenizer(
            text, 
            max_length=self.max_length,  
            truncation=True,
        )
        model_inputs = {k : v[0] for k,v in model_inputs.items()}
        
        num_labels = len(label2id)
        one_hot = np.zeros(num_labels)
        for label in labels.split():
            label_id = label2id[label]
            one_hot[label_id] = 1
        
        model_inputs["labels"] = torch.tensor(one_hot, dtype=torch.float)
        
        return model_inputs


    
