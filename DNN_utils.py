import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(train_df, test_df, lagged_num=5, rolling_window = False):
    '''
    Prepares the train and test data frames by creating lagged returns and encoding 
    categorical variables.
    '''

    # Prepare Data with lagged returns and categorical encoding
    # Sort and add lagged returns per PERMNO
    #daily = daily.sort_values(['PERMNO', 'date'])

    # Shift returns to create lagged features; keep NaNs for first few rows
    for i in range(lagged_num):
        train_df[f'DlyRet_lag{i+1}'] = train_df.groupby('PERMNO')['DlyRet'].shift(i+1)
        train_df[f'DlyRet_lag{i+1}'] = train_df[f'DlyRet_lag{i+1}'].fillna(train_df[f'DlyRet_lag{i+1}'].mean())
        test_df[f'DlyRet_lag{i+1}'] = test_df.groupby('PERMNO')['DlyRet'].shift(i+1)
        test_df[f'DlyRet_lag{i+1}'] = test_df[f'DlyRet_lag{i+1}'].fillna(test_df[f'DlyRet_lag{i+1}'].mean())

    # Construct mean values overe rolling window of 10, 20 and 30 days
    if rolling_window == True:
        for i in range(3):
            train_df[f'DlyRet_roll_{(i+1)*10}'] = train_df.groupby('PERMNO')['DlyRet'].rolling(window = (i+1)*10).mean().fillna(0)
            test_df[f'DlyRet_roll_{(i+1)*10}'] = test_df.groupby('PERMNO')['DlyRet'].rolling(window = (i+1)*10).mean().fillna(0)

    # Encode categorical columns for embeddings
    categorical_columns = ['SICCD']
    for col in categorical_columns:
        train_df[col] = train_df[col].astype('category')
        train_df[col] = train_df[col].cat.codes
        test_df[col] = test_df[col].astype('category')
        test_df[col] = test_df[col].cat.codes

    # Feature lists
    features = [f'DlyRet_lag{i+1}' for i in range(lagged_num)]
    features.append('sprtrn')  

    remove_columns = ['PERMCO', 'year_month', 'NAICS'] # Unused in DNN
    train_df = train_df.drop(remove_columns, axis=1)
    test_df = test_df.drop(remove_columns, axis=1)

    return train_df, test_df, features, categorical_columns

# Dataset class
class FinancialDataset(Dataset):
    def __init__(self, df, features, cat_features, target_col='DlyRet'):
        self.X_num = df[features].values.astype(np.float32)
        self.X_cat = df[cat_features].values.astype(np.int64)
        self.y = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_prob=0.3):  # add dropout param
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout_prob)  # add dropout

        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout_prob)  # add dropout

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)  # apply dropout

        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.act2(out)
        out = self.dropout2(out)  # apply dropout
        return out
class ResidualMLP(nn.Module):
    def __init__(self, num_numeric_feats, cat_dims, embedding_dim=8, hidden_dim=64, n_blocks=3, dropout_prob=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, embedding_dim) for cat_dim in cat_dims
        ])

        input_dim = num_numeric_feats + embedding_dim * len(cat_dims)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.act_in = nn.LeakyReLU(0.1)
        self.dropout_in = nn.Dropout(dropout_prob)  # dropout after initial layer

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_prob) for _ in range(n_blocks)]
        )

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embedded, dim=1)

        x = self.fc_in(x)
        x = self.bn_in(x)
        x = self.act_in(x)
        x = self.dropout_in(x)  # dropout applied here

        x = self.res_blocks(x)
        out = self.fc_out(x).squeeze(-1)
        return out

def sharpe_ratio_loss(y_pred, y_true, eps=1e-6):
    port_returns = y_pred * y_true
    mean = port_returns.mean()
    std = port_returns.std()
    sharpe = mean / (std + eps)
    return -sharpe




