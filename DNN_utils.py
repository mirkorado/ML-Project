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
from sklearn.preprocessing import LabelEncoder


def compute_monthly_sharpe(returns):
    # Multiply by sqrt(21) to get monthly Sharpe ratio
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(21)
    return sharpe_ratio

def compute_sharpe(returns):
    # Multiply by sqrt(252) to get annualized Sharpe ratio
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
    return sharpe_ratio

def progressive_cost(weight_change, base_tc=0.003, alpha=10):
    """
    Penalizes weight changes progressively.
    - base_tc: base transaction cost rate (e.g., 0.001)
    - alpha: aggressiveness of penalty on large trades
    """
    return base_tc * (weight_change + alpha * weight_change**2)


def read_daily_returns(path, nrows=None, skiprows = None, header=None, low_quantile=0.005, up_quantile=0.995):

    # Read data
    daily = pd.read_csv(path, nrows=nrows, skiprows=skiprows, header=header)
    
    # Ensure datetime format for dates and align with predictors data set start date
    daily['date'] = pd.to_datetime(daily['date'], format = "%Y-%m-%d") 
    daily = daily[daily['date'] >= '2000-01-31'] 

    # Remove outliers
    # Compute quantile thresholds for winsorization
    lower_quantile = daily['DlyRet'].quantile(low_quantile)
    upper_quantile = daily['DlyRet'].quantile(up_quantile)

    # Identify outliers for reporting
    outliers = (daily['DlyRet'] < lower_quantile) | (daily['DlyRet'] > upper_quantile)
    print(f"Number of daily return outliers: {outliers.sum():,}")

    # Winsorize: cap values at the quantile thresholds
    daily['DlyRet'] = daily['DlyRet'].clip(lower=lower_quantile, upper=upper_quantile)
    
    # Create lagged return for the S&P 500 index 
    sp500_lagged = daily[['date', 'sprtrn']].drop_duplicates().sort_values('date')
    sp500_lagged['sprtrn_lag1'] = sp500_lagged['sprtrn'].shift(1).fillna(0.0)
    
    # Merge lagged S&P 500 return back into main DataFrame
    daily = daily.merge(sp500_lagged[['date', 'sprtrn_lag1']], on='date', how='left')

    # Encode categorical variables for DNN training
    # Fit once on ALL data (train + test combined)
    #le = LabelEncoder()
    #daily['SICCD'] = le.fit_transform(daily['SICCD'])

    return daily


def prepare_data(train_df, test_df, lagged_num=5, rolling_window = False):
    '''
    Prepares the train and test data frames by creating lagged returns 
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

    # Encode categorical columns for embeddings (already done with whole initial df)
    categorical_columns = ['SICCD']
        
    # Feature lists
    remove_columns = ['PERMCO', 'year_month', 'NAICS', 'date', 'SICCD', 'PERMNO', 'DlyRet', 'sprtrn']  # Unused for training
    features = train_df.columns.tolist()
    features = [col for col in features if col not in remove_columns]
    print(features)
 
    remove_columns_1 = ['PERMCO', 'year_month', 'NAICS']
    train_df = train_df.drop(remove_columns_1, axis=1)
    test_df = test_df.drop(remove_columns_1, axis=1)

    return train_df, test_df, features, categorical_columns

# Dataset class
class FinancialDataset(Dataset):
    """
    A custom PyTorch Dataset for financial tabular data.

    Stores numerical and categorical features along with a target column,
    converting them to appropriate PyTorch-compatible dtypes for memory
    efficiency and training performance.

    Attributes:
        X_num (np.ndarray): Numerical features as float32.
        X_cat (np.ndarray): Categorical features as int32.
        y (np.ndarray): Target values as float32.
    """
    # Use float32 for low memory usage and training speedup 
    def __init__(self, df, features, cat_features, target_col='DlyRet'):
        """
        Initializes the dataset with numerical, categorical features, and the target column.

        Args:
            df (pd.DataFrame): Input dataframe containing the features and target.
            features (List[str]): List of numerical feature column names.
            cat_features (List[str]): List of categorical feature column names.
            target_col (str): Name of the target column. Defaults to 'DlyRet'.
        """
        self.X_num = df[features].values.astype(np.float32)
        self.X_cat = df[cat_features].values.astype(np.int32)
        self.y = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

class ResidualBlock(nn.Module):
    """
    A residual block with two fully connected layers, batch normalization,
    LeakyReLU activations, and dropout.

    This block allows the model to learn identity mappings and improves
    training stability and depth handling.

    Args:
        dim (int): Dimension of the input and output features.
        dropout_prob (float): Probability of dropout. Defaults to 0.3.
    """
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
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
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
    """
    A multilayer perceptron with embedding layers for categorical inputs and 
    residual blocks for deep feature extraction.

    This model combines numerical and embedded categorical features and applies
    multiple residual blocks to learn complex relationships.

    Args:
        num_numeric_feats (int): Number of numerical input features.
        cat_dims (List[int]): List containing the number of categories for each categorical feature.
        embedding_dim (int): Size of embedding vectors for categorical features. Defaults to 8.
        hidden_dim (int): Dimension of hidden layers. Defaults to 64.
        n_blocks (int): Number of residual blocks. Defaults to 3.
        dropout_prob (float): Dropout probability. Defaults to 0.2.
    """
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
        """
        Forward pass through the ResidualMLP model.

        Args:
            x_num (torch.Tensor): Numerical feature tensor of shape (batch_size, num_numeric_feats).
            x_cat (torch.Tensor): Categorical feature tensor of shape (batch_size, num_categorical_feats).

        Returns:
            torch.Tensor: Predicted values of shape (batch_size,).
        """
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


def train_DNN(train_df, test_df, features, cat_features, epochs=50, learning_rate=0.001, 
              max_weight=0.05, diversification_lambda=0.5, temperature=0.3, loss_function = 'softmax_reg'):
    """
    Trains a deep neural network (DNN) using tabular financial data with numerical and categorical features.

    This function sets up data preprocessing, model architecture (ResidualMLP), and training logic 
    using one of several supported loss functions: softmax regression, negative Sharpe ratio, or linear ranking.

    Args:
        train_df (pd.DataFrame): Training dataset containing features and target column.
        test_df (pd.DataFrame): Test dataset containing features and target column.
        features (List[str]): List of numerical feature column names.
        cat_features (List[str]): List of categorical feature column names.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        learning_rate (float, optional): Learning rate for the AdamW optimizer. Defaults to 0.001.
        max_weight (float, optional): Maximum position weight used in softmax-based allocation. Defaults to 0.05.
        diversification_lambda (float, optional): Weight for diversification penalty in softmax loss. Defaults to 0.5.
        temperature (float, optional): Temperature parameter for softmax allocation. Defaults to 0.3.
        loss_function (str, optional): Loss function to use. One of {'softmax_reg', 'neg_sharpe', 'lin_rank'}.
            - 'softmax_reg': Softmax portfolio weighting with diversification regularization.
            - 'neg_sharpe': Negative Sharpe ratio loss.
            - 'lin_rank': Linear rank-based loss focused on top-K selection.

    Returns:
        Tuple:
            train_losses (List[float]): Training loss values over epochs.
            test_losses (List[float]): Test loss values over epochs.
            train_sharpes (List[float]): Training Sharpe ratios over epochs.
            test_sharpes (List[float]): Test Sharpe ratios over epochs.
            strat_returns (List[np.ndarray]): List of daily returns from the strategy on the test set.
            weights (Optional[List[np.ndarray]]): Portfolio weights predicted by the model (only for softmax_reg and lin_rank).
    """
    
    TRAIN_BATCH_SIZE = 2048
    TEST_BATCH_SIZE = 4096

    for col in cat_features:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    
    # Simple data loaders without excessive configuration
    train_dataset = FinancialDataset(train_df, features, cat_features)
    test_dataset = FinancialDataset(test_df, features, cat_features)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # Model setup
    num_numeric_feats = len(features)
    # Compute correct embedding dimensions across train + test
    cat_dims = [
        pd.concat([train_df[col], test_df[col]]).nunique() + 1
        for col in cat_features
    ]

    model = ResidualMLP(num_numeric_feats, cat_dims)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7)

    print(f"Starting training with {len(train_loader)} train batches, {len(test_loader)} test batches")

    if loss_function == 'softmax_reg':
        train_losses, test_losses, train_sharpes, test_sharpes, strat_returns, weights = train_model(
            model, train_loader, test_loader, optimizer, scheduler, epochs=epochs, max_weight=max_weight, 
            diversification_lambda=diversification_lambda, temperature=temperature
        )
    elif loss_function == 'neg_sharpe':
        train_losses, test_losses, train_sharpes, test_sharpes, strat_returns = train_model_negative_sharpe(
            model, train_loader, test_loader, optimizer, scheduler, epochs=epochs
        )
        weights = None
    elif loss_function == 'lin_rank':
        train_losses, test_losses, train_sharpes, test_sharpes, strat_returns, weights = train_model_rank(
            model, train_loader, test_loader, optimizer, scheduler, epochs=epochs, top_k=0.1
        )
    else:
        raise ValueError("Unknown loss function. Implemented loss functions are: softmax_reg ; neg_sharpe ; lin_rank")

    
    print(f"Training completed! Best test Sharpe ratio: {max(test_sharpes):.4f}")
    return train_losses, test_losses, train_sharpes, test_sharpes, strat_returns, weights



def train_model_negative_sharpe(model, train_loader, test_loader, optimizer, scheduler, epochs=50):
    """
    Returns test predictions from the epoch with highest Sharpe ratio as pandas Series
    """
    train_losses = []
    test_losses = []
    train_sharpes = []
    test_sharpes = []

    # Keep track of best predictions
    best_test_sharpe = float('-inf')
    best_test_predictions = None
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        total_loss = 0
        
        for x_num, x_cat, y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(x_num, x_cat)
            loss = neg_sharpe_ratio_loss(outputs, y)
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
                loss = neg_sharpe_ratio_loss(outputs, y)
                total_test_loss += loss.item()
                
                # Store strategy returns for each sample in the batch
                # Remember we use predictions as weights and multiply with the realised returns
                strategy_returns = outputs.cpu().numpy() # Element-wise multiplication
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
            #torch.save(model.state_dict(), 'best_model.pth')
            # Concatenate all batch predictions into a single array
            best_test_predictions = np.concatenate(epoch_test_preds, axis=0)
        
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train fit: {train_sharpe:.4f}, "
                  f"Test fit: {test_sharpe:.4f}")
    
    return (train_losses, test_losses, train_sharpes, test_sharpes, 
            pd.Series(best_test_predictions.flatten()))

def neg_sharpe_ratio_loss(y_pred, y_true, eps=1e-6):
    port_returns = y_pred * y_true
    mean = port_returns.mean()
    std = port_returns.std()
    sharpe = mean / (std + eps)
    return -sharpe

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


def train_model(model, train_loader, test_loader, optimizer, scheduler, epochs=50, max_weight=0.05, diversification_lambda=0.5, temperature=0.3):
    """
    Training loop with diversified Sharpe ratio loss.
    Returns test predictions from the epoch with highest Sharpe ratio.
    """
    train_losses = []
    test_losses = []
    train_sharpes = []
    test_sharpes = []
    
    best_test_sharpe = float('-inf')
    best_test_predictions = None
    best_test_weights = None  # Store weights for analysis
    
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        total_loss = 0
        
        for x_num, x_cat, y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(x_num, x_cat)
            
            # ------ Modified Sharpe Loss ------
            # 1. Apply tempered softmax
            weights = F.softmax(outputs / temperature, dim=0)
            
            # 2. Penalize weights > max_weight
            weight_penalty = torch.sum(F.relu(weights - max_weight))
            
            # 3. Entropy penalty (encourage diversification)
            entropy = -torch.sum(weights * torch.log(weights + 1e-6))
            entropy_penalty = -entropy
            
            # 4. Portfolio returns and Sharpe
            port_returns = weights * y
            mean_return = port_returns.mean()
            std_return = port_returns.std()
            sharpe = mean_return / (std_return + 1e-6)
            
            # 5. Combined loss
            loss = -sharpe + diversification_lambda * (weight_penalty + entropy_penalty)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation phase
        model.eval()
        total_test_loss = 0
        epoch_test_preds = []
        epoch_test_weights = []
        
        with torch.no_grad():
            for x_num, x_cat, y in test_loader:
                outputs = model(x_num, x_cat)
                weights = F.softmax(outputs / temperature, dim=0)
                
                # Compute test loss (same as training)
                weight_penalty = torch.sum(F.relu(weights - max_weight))
                entropy = -torch.sum(weights * torch.log(weights + 1e-6))
                port_returns = weights * y
                sharpe = port_returns.mean() / (port_returns.std() + 1e-6)
                loss = -sharpe + diversification_lambda * (weight_penalty - entropy)
                
                total_test_loss += loss.item()
                epoch_test_preds.append(outputs.cpu().numpy())
                epoch_test_weights.append(weights.cpu().numpy())  # Store weights
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Sharpe evaluation (using actual weighted returns)
        train_sharpe = evaluate_sharpe_diversified(model, train_loader, temperature, max_weight)
        test_sharpe = evaluate_sharpe_diversified(model, test_loader, temperature, max_weight)
        
        train_sharpes.append(train_sharpe)
        test_sharpes.append(test_sharpe)
        
        scheduler.step(test_sharpe)
        
        # Save best model and predictions
        if test_sharpe > best_test_sharpe:
            best_test_sharpe = test_sharpe
            #torch.save(model.state_dict(), 'best_model.pth')
            best_test_predictions = np.concatenate(epoch_test_preds, axis=0)
            best_test_weights = np.concatenate(epoch_test_weights, axis=0)
        
        # Print diagnostics every few epochs
        if epoch % 2 == 0 or epoch == epochs - 1:
            avg_weight = np.mean(best_test_weights) if best_test_weights is not None else 0
            max_w = np.max(best_test_weights) if best_test_weights is not None else 0
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train fit: {train_sharpe:.4f}, "
                  f"Test fit: {test_sharpe:.4f}, "
                  f"Avg Weight: {avg_weight:.4f}, "
                  f"Max Weight: {max_w:.4f}")

    return (
        train_losses, 
        test_losses, 
        train_sharpes, 
        test_sharpes, 
        pd.Series(best_test_predictions.flatten()),
        pd.Series(best_test_weights)  # Return weights for analysis
    )

def evaluate_sharpe_diversified(model, loader, temperature=0.3, max_weight=0.05):
    """Evaluates Sharpe ratio using tempered softmax weights."""
    model.eval()
    port_returns = []
    
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            outputs = model(x_num, x_cat)
            weights = F.softmax(outputs / temperature, dim=0)
            port_returns.append(weights * y)
    
    port_returns = torch.cat(port_returns)
    mean_ret = port_returns.mean()
    std_ret = port_returns.std()
    return (mean_ret / (std_ret + 1e-6)).item()
    
def sharpe_ratio_loss(y_pred, y_true, max_weight=0.05, diversification_lambda=0.5, eps=1e-6):
    """
    Args:
        y_pred: Raw predictions (before softmax)
        y_true: Actual returns
        max_weight: Maximum allowed weight for any single stock (e.g., 5%)
        diversification_lambda: Strength of the diversification penalty 
        eps: Small value to avoid division by zero
    """
    # Apply tempered softmax to get weights (lower temperature = more diversified)
    temperature = 0.3  # Lower = more uniform weights
    weights = F.softmax(y_pred / temperature, dim=0)
    
    # Penalize weights exceeding max_weight
    weight_penalty = torch.sum(F.relu(weights - max_weight))
    
    # Add entropy penalty to encourage diversification (higher entropy = more diversified)
    entropy = -torch.sum(weights * torch.log(weights + eps))
    entropy_penalty = -entropy  # We want to maximize entropy
    
    # Compute portfolio returns and Sharpe ratio
    port_returns = weights * y_true
    mean_return = port_returns.mean()
    std_return = port_returns.std()
    sharpe = mean_return / (std_return + eps)
    
    # Combine Sharpe with penalties
    loss = -sharpe + diversification_lambda * (weight_penalty + entropy_penalty)
    
    return loss



def train_model_rank(model, train_loader, test_loader, optimizer, scheduler, epochs=50, top_k=0.1):
    """
    Training loop with linear rank-based loss.
    Returns test predictions from the epoch with highest Sharpe ratio.
    Args:
        top_k: Fraction of top stocks to select (0.1 = top 10%)
    """
    train_losses = []
    test_losses = []
    train_sharpes = []
    test_sharpes = []
    
    best_test_sharpe = float('-inf')
    best_test_predictions = None
    best_test_weights = None

    for epoch in tqdm(range(epochs), desc="Training"):
        # Training phase
        model.train()
        total_loss = 0
        
        for x_num, x_cat, y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(x_num, x_cat)
            
            # Get smoothed ranks (differentiable approximation)
            ranks = smooth_rank(outputs)
            
            # Select top_k% of stocks
            k = int(len(outputs) * top_k)
            top_mask = ranks >= (len(outputs) - k)
            
            # Assign linear weights based on rank
            weights = torch.zeros_like(outputs)
            if top_mask.sum() > 0:  # Avoid division by zero
                weights[top_mask] = (ranks[top_mask] - (len(outputs) - k) + 1)
                weights = weights / weights.sum()
            
            # Portfolio returns and Sharpe
            port_returns = weights * y
            if weights.sum() > 0:
                mean_return = port_returns.mean()
                std_return = port_returns.std()
                sharpe = mean_return / (std_return + 1e-6)
                loss = -sharpe
            else:
                # Force a dummy tensor with requires_grad=True to keep graph alive
                loss = torch.tensor(0.0, requires_grad=True, device=outputs.device)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation phase
        model.eval()
        total_test_loss = 0
        epoch_test_preds = []
        epoch_test_weights = []
        
        with torch.no_grad():
            for x_num, x_cat, y in test_loader:
                outputs = model(x_num, x_cat)
                ranks = smooth_rank(outputs)
                k = int(len(outputs) * top_k)
                top_mask = ranks >= (len(outputs) - k)
                
                weights = torch.zeros_like(outputs)
                if top_mask.sum() > 0:
                    weights[top_mask] = (ranks[top_mask] - (len(outputs) - k) + 1)
                    weights = weights / weights.sum()
                
                # Compute test loss
                port_returns = weights * y
                sharpe = port_returns.mean() / (port_returns.std() + 1e-6)
                loss = -sharpe
                
                total_test_loss += loss.item()
                epoch_test_preds.append(outputs.cpu().numpy())
                epoch_test_weights.append(weights.cpu().numpy())
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Sharpe evaluation
        train_sharpe = evaluate_rank_sharpe(model, train_loader, top_k)
        test_sharpe = evaluate_rank_sharpe(model, test_loader, top_k)
        
        train_sharpes.append(train_sharpe)
        test_sharpes.append(test_sharpe)
        
        scheduler.step(test_sharpe)
        
        # Save best model and predictions
        if test_sharpe > best_test_sharpe:
            best_test_sharpe = test_sharpe
            #torch.save(model.state_dict(), 'best_model.pth')
            best_test_predictions = np.concatenate(epoch_test_preds, axis=0)
            best_test_weights = np.concatenate(epoch_test_weights, axis=0)
        
        # Print diagnostics
        if epoch % 2 == 0 or epoch == epochs - 1:
            avg_weight = np.mean(best_test_weights) if best_test_weights is not None else 0
            max_w = np.max(best_test_weights) if best_test_weights is not None else 0
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train fit: {train_sharpe:.4f}, "
                  f"Test fit: {test_sharpe:.4f}, "
                  f"Avg Weight: {avg_weight:.4f}, "
                  f"Max Weight: {max_w:.4f}")

    return (
        train_losses, 
        test_losses, 
        train_sharpes, 
        test_sharpes, 
        pd.Series(best_test_predictions.flatten()),
        pd.Series(best_test_weights.flatten()) if best_test_weights is not None else None
    )


def smooth_rank(x, alpha=0.01):
    """Differentiable approximation of ranks using sigmoid.
    Args:
        x: Tensor of raw predictions
        alpha: Smoothing parameter (smaller = sharper ranking)
    Returns:
        Approximate ranks (higher values = higher rank)
    """
    pairwise_diff = x.unsqueeze(1) - x.unsqueeze(0)
    ranks = torch.sigmoid(pairwise_diff / alpha).sum(dim=1)
    return ranks


def evaluate_rank_sharpe(model, loader, top_k):
    """Evaluate Sharpe ratio using rank-based weights."""
    model.eval()
    port_returns = []
    
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            outputs = model(x_num, x_cat)
            ranks = smooth_rank(outputs) # Use smooth ranks for differentiability
            k = int(len(outputs) * top_k)
            top_mask = ranks >= (len(outputs) - k)
            
            weights = torch.zeros_like(outputs)
            if top_mask.sum() > 0:
                weights[top_mask] = (ranks[top_mask] - (len(outputs) - k) + 1)
                weights = weights / weights.sum()
            
            port_returns.append(weights * y)
    
    port_returns = torch.cat(port_returns)
    mean_ret = port_returns.mean()
    std_ret = port_returns.std()
    return (mean_ret / (std_ret + 1e-6)).item()


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



# Process each PERMNO individually (more memory efficient)
def memory_efficient_merge(monthly_df, daily_df):
    """
    Process each PERMNO individually to ensure proper sorting.
    
    """
    # Step 1: Pre-process data
    monthly = monthly_df[['date', 'PERMNO', 'pls_index']].copy()
    monthly['date'] = pd.to_datetime(monthly['date'])
    monthly['PERMNO'] = monthly['PERMNO'].astype('int32')
    
    # Filter daily data
    valid_permnos = monthly['PERMNO'].unique()
    daily_df = daily_df[daily_df['PERMNO'].isin(valid_permnos)].copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df['PERMNO'] = daily_df['PERMNO'].astype('int32')
    
    # Step 2: Group by PERMNO and process each group
    results = []
    
    for permno in tqdm(valid_permnos, desc="Processing PERMNOs"):
        # Get data for this PERMNO
        daily_stock = daily_df[daily_df['PERMNO'] == permno].sort_values('date')
        monthly_stock = monthly[monthly['PERMNO'] == permno].sort_values('date')
        
        if len(daily_stock) == 0 or len(monthly_stock) == 0:
            continue
            
        # Merge for this stock
        merged_stock = pd.merge_asof(
            daily_stock,
            monthly_stock,
            on='date',
            direction='forward'
        )
        
        # Drop rows without index values
        merged_stock = merged_stock.dropna(subset=['pls_index'])
        
        if len(merged_stock) > 0:
            results.append(merged_stock)
    
    # Combine all results
    if results:
        result = pd.concat(results, ignore_index=True)
    else:
        result = pd.DataFrame()
    
    return result

