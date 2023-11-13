import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import Counter
from torch.utils.data import Dataset, DataLoader

NUM_TAGS = 256

class TaggingDataset(Dataset):
    def __init__(self, df, track_idx2embeds, aug=0, testing=False, label_smoothing = 0):
        self.df = df
        self.testing = testing
        self.aug = aug
        self.track_idx2embeds = track_idx2embeds
        self.label_smoothing = label_smoothing
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_idx = row.track
        embeds = self.track_idx2embeds[track_idx]
        if self.testing:
            return track_idx, embeds
        tags = [int(x) for x in row.tags.split(',')]
        target = np.zeros(NUM_TAGS)
        target[tags] = 1
        if self.label_smoothing > 0:
            eps = self.label_smoothing / 256
            target = target * (1 - self.label_smoothing) + eps
        
        if np.random.choice([0, 1], p=[1 - self.aug, self.aug]):
            s = np.random.uniform(0.0, 0.4)
            e = np.random.uniform(s+0.1, 1)
            s = int(s * embeds.shape[0])
            e = int(e * embeds.shape[0])
            embeds = embeds[s:e]
        
        return track_idx, embeds, target
    
def collate_fn(b):
    track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
    embeds = [torch.from_numpy(x[1]) for x in b]
    targets = np.vstack([x[2] for x in b])
    targets = torch.from_numpy(targets)
    return track_idxs, embeds, targets

def collate_fn_test(b):
    track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
    embeds = [torch.from_numpy(x[1]) for x in b]
    return track_idxs, embeds

def get_train_probs(df_train):
    tags = [[int(i) for i in x.split(',')] for x in df_train.tags.values]
    dict_tags = {}
    for cls_tags in tags:
        for c in cls_tags:
            if c not in dict_tags.keys():
                dict_tags[c] = Counter(cls_tags)
            else:
                dict_tags[c].update(Counter(cls_tags))

    for tag in dict_tags.keys():
        del dict_tags[tag][tag]
        n = np.sum(list(dict_tags[tag].values()))
        for t in dict_tags[tag].keys():
            dict_tags[tag][t] = dict_tags[tag][t]/n
    return dict_tags

def load_data():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    track_idx2embeds = {}
    for fn in tqdm(glob('track_embeddings/*')):
        name = fn.split('/')[1].split('.')[0]
        if name == "track_embeddings":
            continue
        track_idx = int(name)
        embeds = np.load(fn)
        track_idx2embeds[track_idx] = embeds
        
    return df_train, df_test, track_idx2embeds