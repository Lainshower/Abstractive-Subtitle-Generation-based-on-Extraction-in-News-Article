import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, IterableDataset

class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tok, max_len, pad_index = None, ignore_index=-100):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.docs = pd.read_csv(file)
        self.len = self.docs.shape[0]
        if pad_index is None:
            self.pad_index = self.tok.pad_token_id
        else:
            self.pad_index = pad_index
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs, max_len):
        if len(inputs) < max_len:
            pad = np.array([self.pad_index] *(max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:max_len]

        return inputs

    def add_ignored_data(self, inputs, max_len):
        if len(inputs) < max_len:
            pad = np.array([self.ignore_index] *(max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tok.encode(instance['extracted_body'])
        input_ids = self.add_padding_data(input_ids, max_len=self.max_len)

        label_ids = self.tok.encode(instance['subtitle'])
        label_ids.append(self.tok.eos_token_id)
        dec_input_ids = [self.tok.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids, max_len=128)
        label_ids = self.add_ignored_data(label_ids, max_len=128)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}
    
    def __len__(self):
        return self.len