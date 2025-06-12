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
        train_df[f'DlyRet_lag{i+1}'] = train_df.groupby('PERMNO')['DlyRet'].shift(i+1).fillna(0.0)
        test_df[f'DlyRet_lag{i+1}'] = test_df.groupby('PERMNO')['DlyRet'].shift(i+1).fillna(0.0)

    if rolling_window == True:
        for i in range(3):
            window = (i + 1) * 10
            train_df[f'DlyRet_roll_{window}'] = (
                train_df
                .groupby('PERMNO')['DlyRet_lag1']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().fillna(0.0))
            )

            test_df[f'DlyRet_roll_{window}'] = (
                test_df
                .groupby('PERMNO')['DlyRet_lag1']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean().fillna(0.0))
            )

    # Encode categorical columns for embeddings
    categorical_columns = ['SICCD']
    for col in categorical_columns:
        train_df[col] = train_df[col].astype('category')
        train_df[col] = train_df[col].cat.codes
        test_df[col] = test_df[col].astype('category')
        test_df[col] = test_df[col].cat.codes
    
    # Feature lists
    remove_columns = ['PERMCO', 'year_month', 'NAICS', 'date', 'SICCD', 'PERMNO', 'DlyRet']  # Unused for training
    features = train_df.columns.tolist()
    features = [col for col in features if col not in remove_columns]
    print(features)
 
    remove_columns_1 = ['PERMCO', 'year_month', 'NAICS']
    train_df = train_df.drop(remove_columns_1, axis=1)
    test_df = test_df.drop(remove_columns_1, axis=1)

    return train_df, test_df, features, categorical_columns

# Dataset class
class FinancialDataset(Dataset):
    # Use float32 for low memory usage and training speedup 
    def __init__(self, df, features, cat_features, target_col='DlyRet'):
        self.X_num = df[features].values.astype(np.float32)
        self.X_cat = df[cat_features].values.astype(np.int32)
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


def train_DNN(train_df, test_df, features, cat_features, epochs=50, learning_rate=0.001):
    
    TRAIN_BATCH_SIZE = 2048
    TEST_BATCH_SIZE = 4096

    # Simple data loaders without excessive configuration
    train_dataset = FinancialDataset(train_df, features, cat_features)
    test_dataset = FinancialDataset(test_df, features, cat_features)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # Model setup
    num_numeric_feats = len(features)
    cat_dims = [train_df[col].nunique() for col in cat_features]

    model = ResidualMLP(num_numeric_feats, cat_dims)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7)

    print(f"Starting training with {len(train_loader)} train batches, {len(test_loader)} test batches")

    train_losses, test_losses, train_sharpes, test_sharpes, strat_returns = train_model(
        model, train_loader, test_loader, optimizer, scheduler, epochs=epochs
    )

    print(f"Training completed! Best test Sharpe ratio: {max(test_sharpes):.4f}")
    return train_losses, test_losses, train_sharpes, test_sharpes, strat_returns


def train_model(model, train_loader, test_loader, optimizer, scheduler, epochs=50):
    """
    Simplified training loop without unnecessary optimizations
    Returns test predictions from the epoch with highest Sharpe ratio as pandas Series
    """
    train_losses = []
    test_losses = []
    train_sharpes = []
    test_sharpes = []
    
    best_test_sharpe = float('-inf')
    best_test_predictions = None
    
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
        epoch_test_preds = []
        
        with torch.no_grad():
            for x_num, x_cat, y in test_loader:
                outputs = model(x_num, x_cat)
                loss = sharpe_ratio_loss(outputs, y)
                total_test_loss += loss.item()
                
                # Store strategy returns for each sample in the batch
                # Remember we use predictions as weights and multiply with the realised returns
                strategy_returns = outputs.cpu().numpy() #* y.cpu().numpy()  # Element-wise multiplication
                epoch_test_preds.append(strategy_returns)  # Append the array
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Sharpe evaluation
        train_sharpe = evaluate_sharpe(model, train_loader)
        test_sharpe = evaluate_sharpe(model, test_loader)
        
        train_sharpes.append(train_sharpe)
        test_sharpes.append(test_sharpe)
        
        scheduler.step(test_sharpe)
        
        # Save best model and predictions
        if test_sharpe > best_test_sharpe:
            best_test_sharpe = test_sharpe
            torch.save(model.state_dict(), 'best_model.pth')
            # Concatenate all batch predictions into a single array
            best_test_predictions = np.concatenate(epoch_test_preds, axis=0)
        
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train Sharpe: {train_sharpe:.4f}, "
                  f"Test Sharpe: {test_sharpe:.4f}")
    
    return (train_losses, test_losses, train_sharpes, test_sharpes, 
            pd.Series(best_test_predictions.flatten()))

def evaluate_sharpe(model, loader):
    """Evaluate Sharpe ratio on a given data loader.
    
    Args:
        model: PyTorch model
        loader: DataLoader containing batches of (x_num, x_cat, y)
        
    Returns:
        Sharpe ratio (float)
    """
    model.eval()
    port_returns = []
    
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            preds = model(x_num, x_cat)
            port_returns.append(preds * y)
    
    port_returns = torch.cat(port_returns)
    mean_ret = port_returns.mean()
    std_ret = port_returns.std()
    sharpe = mean_ret / (std_ret + 1e-8)  # Small epsilon to avoid division by zero
    
    return sharpe.item()

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

def merge_pls_with_asof(daily_returns, pls_data):
    """
    Merge PLS index data to daily returns using merge_asof with forward filling.
    
    This function handles the sorting requirements properly and processes each
    PERMNO separately to avoid sorting issues with merge_asof.
    
    Parameters:
    -----------
    daily_returns : DataFrame
        Daily returns data with columns: date, PERMNO, DlyRet, etc.
    pls_data : DataFrame  
        Monthly PLS index data with columns: date, PERMNO, pls_index
    
    Returns:
    --------
    DataFrame
        Merged dataframe with forward-filled pls_index values
    """
    
    # Make copies to avoid modifying original data
    daily_df = daily_returns.copy()
    pls_df = pls_data.copy()
    
    # Ensure date columns are datetime
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    pls_df['date'] = pd.to_datetime(pls_df['date'])
    
    # Get unique PERMNOs from both datasets
    daily_permnos = set(daily_df['PERMNO'].unique())
    pls_permnos = set(pls_df['PERMNO'].unique())
    common_permnos = daily_permnos & pls_permnos
    
    print(f"Processing {len(common_permnos)} common PERMNOs...")
    
    # Process each PERMNO separately to ensure proper sorting
    merged_parts = []
    
    for i, permno in enumerate(common_permnos):
        if i % 1000 == 0:  # Progress indicator
            print(f"Processing PERMNO {i+1}/{len(common_permnos)}")
        
        # Get data for this PERMNO and sort by date
        daily_subset = daily_df[daily_df['PERMNO'] == permno].sort_values('date')
        pls_subset = pls_df[pls_df['PERMNO'] == permno].sort_values('date')
        
        if len(pls_subset) > 0:
            # Perform merge_asof for this PERMNO
            merged_subset = pd.merge_asof(
                daily_subset,
                pls_subset[['date', 'pls_index']],
                on='date',
                direction='backward'  # Forward fill: use most recent available value
            )
            merged_parts.append(merged_subset)
    
    # Handle PERMNOs that exist in daily data but not in PLS data
    permnos_without_pls = daily_permnos - pls_permnos
    if permnos_without_pls:
        print(f"Adding {len(permnos_without_pls)} PERMNOs without PLS data...")
        for permno in permnos_without_pls:
            daily_subset = daily_df[daily_df['PERMNO'] == permno].copy()
            daily_subset['pls_index'] = np.nan
            merged_parts.append(daily_subset)
    
    # Combine all parts
    print("Combining results...")
    merged_df = pd.concat(merged_parts, ignore_index=True)
    
    # Restore original order if needed
    merged_df = merged_df.sort_values(['date', 'PERMNO']).reset_index(drop=True)
    
    return merged_df

def check_merge_quality(merged_df, original_daily_df):
    """
    Check the quality of the merge operation.
    
    Parameters:
    -----------
    merged_df : DataFrame
        Result from merge operation
    original_daily_df : DataFrame
        Original daily returns dataframe
    """
    
    print("\n" + "="*50)
    print("MERGE QUALITY REPORT")
    print("="*50)
    
    print(f"Original daily rows: {len(original_daily_df):,}")
    print(f"Merged rows: {len(merged_df):,}")
    print(f"Rows with PLS data: {merged_df['pls_index'].notna().sum():,}")
    print(f"Rows missing PLS data: {merged_df['pls_index'].isna().sum():,}")
    print(f"Coverage: {(merged_df['pls_index'].notna().sum() / len(merged_df)) * 100:.1f}%")
    
    # Check for duplicates
    duplicates = merged_df.duplicated(subset=['date', 'PERMNO']).sum()
    print(f"Duplicate (date, PERMNO) pairs: {duplicates}")
    
    # Sample check - show forward fill working
    print(f"\nSample forward fill check:")
    sample_permno = merged_df[merged_df['pls_index'].notna()]['PERMNO'].iloc[0]
    sample_data = merged_df[merged_df['PERMNO'] == sample_permno].head(10)
    print(sample_data[['date', 'PERMNO', 'DlyRet', 'pls_index']].to_string())
    
    # Check date range coverage
    print(f"\nDate range:")
    print(f"Daily data: {merged_df['date'].min()} to {merged_df['date'].max()}")
    pls_dates = merged_df[merged_df['pls_index'].notna()]['date']
    if len(pls_dates) > 0:
        print(f"PLS data: {pls_dates.min()} to {pls_dates.max()}")


# Quick test function to verify the logic
def test_merge_logic():
    """
    Test the merge logic with sample data
    """
    # Create sample data
    daily_sample = pd.DataFrame({
        'date': pd.date_range('2000-01-01', '2000-01-31', freq='D'),
        'PERMNO': [10001] * 31,
        'DlyRet': np.random.normal(0, 0.02, 31)
    })
    
    pls_sample = pd.DataFrame({
        'date': ['2000-01-01', '2000-01-15'],
        'PERMNO': [10001, 10001],
        'pls_index': [-1.5, -2.0]
    })
    pls_sample['date'] = pd.to_datetime(pls_sample['date'])
    
    # Test merge
    result = merge_pls_with_asof(daily_sample, pls_sample)
    
    print("Test Results:")
    print(result[['date', 'PERMNO', 'DlyRet', 'pls_index']].head(20))
    
    return result



