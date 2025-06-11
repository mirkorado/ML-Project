import torch
import numpy as np
import pandas as pd
import seaborn as sns
import gc
import os
import matplotlib.pyplot as plt
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

    if rolling_window == True:
        for i in range(3):
            window = (i + 1) * 10
            train_df[f'DlyRet_roll_{window}'] = (
                train_df
                .groupby('PERMNO')['DlyRet']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

            test_df[f'DlyRet_roll_{window}'] = (
                test_df
                .groupby('PERMNO')['DlyRet']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

    # Encode categorical columns for embeddings
    categorical_columns = ['SICCD']
    for col in categorical_columns:
        train_df[col] = train_df[col].astype('category')
        train_df[col] = train_df[col].cat.codes
        test_df[col] = test_df[col].astype('category')
        test_df[col] = test_df[col].cat.codes

    # Feature lists
    remove_columns = ['PERMCO', 'year_month', 'NAICS', 'date', 'SICCD', 'PERMNO']  # Unused in DNN
    features = train_df.columns.tolist()
    features = [col for col in features if col not in remove_columns]
    print(features)
 
    remove_columns_1 = ['PERMCO', 'year_month', 'NAICS']
    train_df = train_df.drop(remove_columns_1, axis=1)
    test_df = test_df.drop(remove_columns_1, axis=1)

    return train_df, test_df, features, categorical_columns

# Dataset class
class FinancialDataset(Dataset):
    # Use float16 for low memory usage and training speedup 
    def __init__(self, df, features, cat_features, target_col='DlyRet'):
        self.X_num = df[features].values.astype(np.float16)
        self.X_cat = df[cat_features].values.astype(np.int32)
        self.y = df[target_col].values.astype(np.float16)

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

# ---- Evaluation function ----
def evaluate_sharpe(model, loader, device=None):
    """Evaluate Sharpe ratio on a given data loader.

    Args:
        model: PyTorch model
        loader: DataLoader containing batches of (x_num, x_cat, y)
        device: Device to run computations on (default: model's device)

    Returns:
        Sharpe ratio (float)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds = []
    all_y = []

    with torch.no_grad():
        for x_num, x_cat, y in loader:
            # Move data to the same device as model
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            preds = model(x_num, x_cat)
            all_preds.append(preds)
            all_y.append(y)

    # Concatenate while keeping tensors 
    preds = torch.cat(all_preds)
    y_true = torch.cat(all_y)

    # Compute portfolio returns
    port_returns = preds * y_true
    mean_ret = port_returns.mean()
    std_ret = port_returns.std()
    sharpe = (mean_ret / (std_ret + 1e-8)).item()  # Convert to Python float

    return sharpe


def train_DNN(train_df, test_df, features, cat_features, epochs=50, learning_rate=0.001):
    
    TRAIN_BATCH_SIZE = 2048
    TEST_BATCH_SIZE = 4096

    # Simple data loaders without excessive configuration
    train_dataset = FinancialDataset(train_df, features, cat_features)
    test_dataset = FinancialDataset(test_df, features, cat_features)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # Model setup
    num_numeric_feats = len(features)
    cat_dims = [train_df[col].nunique() for col in cat_features]

    model = ResidualMLP(num_numeric_feats, cat_dims)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7)

    print(f"Starting training with {len(train_loader)} train batches, {len(test_loader)} test batches")

    train_losses, test_losses, train_sharpes, test_sharpes = train_model(
        model, train_loader, test_loader, optimizer, scheduler, epochs=epochs
    )

    print(f"Training completed! Best test Sharpe ratio: {max(test_sharpes):.4f}")
    return train_losses, test_losses, train_sharpes, test_sharpes


def train_model(model, train_loader, test_loader, optimizer, scheduler, epochs=50):
    """
    Simplified training loop without unnecessary optimizations
    """
    train_losses = []
    test_losses = []
    train_sharpes = []
    test_sharpes = []
    
    best_test_sharpe = float('-inf')
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        total_loss = 0
        
        for x_num, x_cat, y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(x_num, x_cat)
            loss = sharpe_ratio_loss(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation phase
        model.eval()
        total_test_loss = 0
        
        with torch.no_grad():
            for x_num, x_cat, y in test_loader:
                outputs = model(x_num, x_cat)
                loss = sharpe_ratio_loss(outputs, y)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Sharpe evaluation
        train_sharpe = evaluate_sharpe(model, train_loader)
        test_sharpe = evaluate_sharpe(model, test_loader)
        
        train_sharpes.append(train_sharpe)
        test_sharpes.append(test_sharpe)
        
        scheduler.step(test_sharpe)
        
        # Save best model
        if test_sharpe > best_test_sharpe:
            best_test_sharpe = test_sharpe
            torch.save(model.state_dict(), 'best_model.pth')
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train Sharpe: {train_sharpe:.4f}, "
                  f"Test Sharpe: {test_sharpe:.4f}")
    
    return train_losses, test_losses, train_sharpes, test_sharpes


def evaluate_sharpe(model, loader):
    """
    Simplified Sharpe ratio evaluation
    """
    model.eval()
    all_returns = []
    
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            preds = model(x_num, x_cat)
            port_returns = preds * y
            all_returns.append(port_returns)
    
    all_returns = torch.cat(all_returns, dim=0)
    mean_ret = torch.mean(all_returns)
    std_ret = torch.std(all_returns)
    sharpe = mean_ret / (std_ret + 1e-8)
    
    return float(sharpe)

def plot_train_vs_test(epochs, train_losses, test_losses):
  plt.plot(epochs, train_losses, label='Train Loss (neg Sharpe)')
  plt.plot(epochs, test_losses, label='Test Loss (neg Sharpe)')
  plt.xlabel('Epoch')
  plt.ylabel('Loss (Negative Sharpe)')
  plt.title('Train vs Test Loss over Epochs')
  plt.legend()
  plt.grid(True)
  plt.show()

def plot_results_1(epochs, train_losses, train_sharpes, test_sharpes):
  # Plot results
  sns.set(style="darkgrid")
  plt.figure(figsize=(14, 6))

  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_losses, marker='o', color='blue')
  plt.title('Training Loss (Negative Sharpe) Over Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Loss (Negative Sharpe)')
  plt.xticks(ticks=range(0, len(epochs), 5), rotation=45)  # Show ticks every 5 epochs, rotated

  # Plot Sharpe ratios
  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_sharpes, label='Train Sharpe', marker='o', color='green')
  plt.plot(epochs, test_sharpes, label='Test Sharpe', marker='x', color='red')
  plt.title('Train vs Test Sharpe Ratio Over Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Sharpe Ratio')
  plt.legend()
  plt.xticks(ticks=range(0, len(epochs), 5), rotation=45)  # Same here

  plt.tight_layout()
  plt.show()



