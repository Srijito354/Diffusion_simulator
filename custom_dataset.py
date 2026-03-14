import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, data_dir, max_len = 32):
        self.max_len = max_len
        self.df = pd.read_csv(data_dir, sep = "\t")
        #self.coordinates = self.df[['x', 'y']].values
        self.labels = list(set(self.df['dataset']))

        self.label_point_dict = {}

        for label in self.labels:
            points = list(self.df[self.df['dataset'] == label][['x', 'y']].values)
            self.label_point_dict[label] = points

        all_points = np.concatenate(list(self.label_point_dict.values()), axis = 0)
        self.mean = all_points.mean(axis = 0)
        self.std = all_points.std(axis = 0)

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        #print(self.df[['x','y']].mean())
        #print(self.df[['x','y']].std())

    def __len__(self):
        return len(set(self.df["dataset"]))*400

    def __getitem__(self, idx):
        #coordinate = self.coordinates[idx]
        label = self.labels[idx % len(self.labels)]

        points_np = np.array(self.label_point_dict[label])
        points_normalized = (points_np - self.mean)/self.std
        coordinate_tensor = torch.tensor(points_normalized, dtype = torch.float32)

        tokens = self.tokenizer(
            label,
            padding = "max_length",
            max_length = self.max_len,
            truncation = True,
            return_tensors = "pt"
        )

        return {"Coordinate":coordinate_tensor, "input_ids":tokens['input_ids'].squeeze(0), "attention_mask":tokens['attention_mask'].squeeze(0)}

'''
# Some test code.

data_dir = "Datashape.tsv"

dataset = Data(data_dir)
dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

for batch in dataloader:
    print(batch['Coordinate'].shape)
    print(batch['input_ids'].shape)
    break
'''