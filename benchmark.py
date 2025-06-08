import pandas as pd
import numpy as np
from datetime import datetime
import gc
from tqdm import tqdm


def merge_all_data_frames(daily_data, firm_characteristics, jkp_factors, linking_table):
    """
    Merges daily returns, firm characteristics, and JKP factor returns into a single unified DataFrame.

    This function:
    - Merges firm-level characteristics (typically quarterly) with a CRSP-Compustat linking table (indexed by 'gvkey')
      that provides PERMNO and PERMCO mappings.
    - Filters the linking table for valid link types and date ranges.
    - Uses `merge_asof` to forward-fill quarterly firm characteristics to match daily frequency.
    - Merges in JKP factor returns (wide format, indexed by month-end 'date' and with one column per factor).
    - Drops temporary or redundant merge columns and sets the final index for analysis.

    Parameters:
    ----------
    daily_data : pd.DataFrame
        DataFrame of daily stock return data. Must contain columns: ['date', 'PERMNO', 'PERMCO'].

    firm_characteristics : pd.DataFrame
        DataFrame of firm-level characteristics at quarterly frequency.
        Must be indexed by ['date', 'gvkey'].

    jkp_factors : pd.DataFrame
        DataFrame of JKP factor returns in wide format, indexed by month-end dates.
        Each column corresponds to a different factor name, and values are the respective returns.

    linking_table : pd.DataFrame
        Compustat-CRSP linking table indexed by 'gvkey'. Must contain columns: ['PERMNO', 'PERMCO', 'LINKTYPE', 'LINKDT', 'LINKENDDT'].

    Returns:
    -------
    pd.DataFrame
        A merged DataFrame indexed by ['date', 'PERMCO', 'PERMNO'], containing:
        - daily returns,
        - forward-filled firm characteristics (from quarterly),
        - monthly JKP factor returns (aligned to the daily data by month-end).
    """

    # Reset indexes for merging
    firm_characteristics = firm_characteristics.reset_index()
    linking_table = linking_table.reset_index()

    # Merge firm characteristics with linking table
    comp_linked = pd.merge(firm_characteristics, linking_table, how='left', on='gvkey')

    # Filter for valid link types and valid link dates
    comp_linked = comp_linked[comp_linked['LINKTYPE'].isin(['LU', 'LC'])]
    comp_linked = comp_linked[
        (comp_linked['date'] >= comp_linked['LINKDT']) &
        (comp_linked['date'] <= comp_linked['LINKENDDT'])
    ]

    # Drop PERMCO from comp_linked to avoid duplication
    comp_linked = comp_linked.drop(columns=['PERMCO'], errors='ignore')

    # Rename for clarity and compatibility
    comp_linked.rename(columns={'date': 'quarter_date'}, inplace=True)
    comp_linked['PERMNO'] = comp_linked['PERMNO'].astype('int64')

    # Reset index on daily_data for merge_asof
    daily_data = daily_data.reset_index()

    # Merge using merge_asof to align firm characteristics to most recent available quarter
    merged = pd.merge_asof(
        daily_data.sort_values('date'),
        comp_linked.sort_values('quarter_date'),
        by='PERMNO',
        left_on='date',
        right_on='quarter_date',
        direction='backward'
    )
    # Create month-end column to align with JKP factor data
    merged['month_end'] = merged['date'] + pd.offsets.MonthEnd(0)

    if jkp_factors is not None:
        print("Start joining JKP factors")
        # Reset index on jkp_factors to merge
        jkp_factors = jkp_factors.reset_index()

        # Merge JKP factors by month-end date
        merged = pd.merge(merged, jkp_factors, how='left', left_on='month_end', right_on='date')
        print("Finished second merge")

    # Drop redundant or intermediary columns
    merged.drop(columns=[
        'quarter_date', 'month_end', 'LINKDT', 'LINKENDDT', 'LINKTYPE',
        'gvkey', 'date_y'
    ], inplace=True, errors='ignore')

    # Rename date_x back to date for clarity
    merged.rename(columns={'date_x': 'date'}, inplace=True)

    # Set final index as required
    merged.set_index(['date', 'PERMCO', 'PERMNO'], inplace=True)

    return merged

def slice_dataframes(dailyret: pd.DataFrame, firmchar: pd.DataFrame, min_date: str, max_date: str, is_training: bool = True) -> tuple:
    """
    Slices two dataframes based on a date range.

    Args:
        dailyret (pd.DataFrame): DataFrame with daily data and 'date' column.
        firmchar (pd.DataFrame): DataFrame with monthly data and 'date' column.
        min_date (str): Start date for slicing (inclusive).
        max_date (str): End date for slicing (inclusive for training, inclusive for testing).
        is_training (bool): Flag to indicate if slicing is for training (True) or testing (False).

    Returns:
        tuple: Sliced dailyret and firmchar DataFrames.
    """
    if is_training:
        # For training: include both min_date and max_date
        dailyret_sliced = dailyret[(dailyret['date'] >= min_date) & (dailyret['date'] <= max_date)].copy()
        firmchar_sliced = firmchar[(firmchar['date'] >= min_date) & (firmchar['date'] <= max_date)].copy()
    else:
        # For testing: start forecasting for the next day after training, BUT we assume that we know the latest update of the factors as of EoM
        dailyret_sliced = dailyret[(dailyret['date'] > min_date) & (dailyret['date'] < max_date)].copy()
        firmchar_sliced = firmchar[(firmchar['date'] >= min_date) & (firmchar['date'] < max_date)].copy()
    return dailyret_sliced, firmchar_sliced

def process_dataframe(df: pd.DataFrame, nan_threshold: float = 0.3) -> tuple:
    """
    Processes a DataFrame by dropping columns with high NaN percentage and filling NaNs with cross-sectional means.

    Args:
        df (pd.DataFrame): Input DataFrame.
        nan_threshold (float): Threshold for dropping columns based on NaN percentage (default: 0.3).

    Returns:
        tuple: Processed DataFrame and dictionary of cross-sectional means.
    """
    # Calculate NaN percentage per column
    nan_pct = df.isna().mean()
    cols_to_drop = nan_pct[nan_pct > nan_threshold].index
    df_processed = df.drop(columns=cols_to_drop)

    # Calculate cross-sectional means (excluding 'date', 'gvkey', 'DlyRet')
    exclude_cols = ['date', 'gvkey', 'DlyRet', 'PERMCO',  'PERMNO',  'SICCD', 'NAICS']
    feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
    cs_means = df_processed[feature_cols].mean()

    # Fill NaNs with cross-sectional means
    for col in feature_cols:
        df_processed[col] = df_processed[col].fillna(cs_means[col])

    return df_processed, cs_means, cols_to_drop

def normalize_weights(predictions: pd.Series) -> pd.Series:
    """
    Applies threshold to predictions and normalizes them to sum to 1 within each date.

    Args:
        predictions (pd.Series): Predicted returns with 'date' and 'PERMCO' and 'PERMNO' in index.

    Returns:
        pd.Series: Normalized weights.
    """

    # Normalize within each date
    weights = predictions.groupby('date')[['prediction']].transform(lambda x: x / x.sum() if x.sum() != 0 else 0)
    return weights
    
def sharpe_ratio(returns):
    return np.round(np.sqrt(250) * returns.mean() / returns.std(), 2)
    

def train_ridge_regr(train_df: pd.DataFrame,
                shrinkage_list: int):
    """
    Regression is
    beta = (zI + S'S/t)^{-1}S'y/t = S' (zI+SS'/t)^{-1}y/t
    Inverting matrices is costly, so we use eigenvalue decomposition:
    (zI+A)^{-1} = U (zI+D)^{-1} U' where UDU' = A is eigenvalue decomposition,
    and we use the fact that D @ B = (diag(D) * B) for diagonal D, which saves a lot of compute cost
    """
    feature_cols = [col for col in train_df.columns if col not in ['date', 'gvkey', 'DlyRet', 'PERMCO',
                                                                   'PERMNO',  'SICCD', 'NAICS' ]]
    signals = train_df[feature_cols]
    labels = train_df['DlyRet'].values

    t_ = signals.shape[0]
    p_ = signals.shape[1]
    if p_ < t_:
        # this is standard regression
        eigenvalues, eigenvectors = np.linalg.eigh(signals.T @ signals / t_)
        means = signals.T @ labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)
        betas = eigenvectors @ intermed
    else:
        # this is the weird over-parametrized regime
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T / t_)
        means = labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means # this is \mu

        # now we build [(z_1+\delta)^{-1}, \cdots, (z_K+\delta)^{-1}] * \mu
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)

        tmp = eigenvectors.T @ signals # U.T @ S
        betas = tmp.T @ intermed # (S.T @ U) @ [(z_1+\delta)^{-1}, \cdots, (z_K+\delta)^{-1}] * \mu
    return betas, feature_cols
    
def predict_ridge_regr(betas: np.ndarray,
                future_signals: np.ndarray):
    predictions = future_signals @ betas
    return predictions
    
    
def portfolio_returns_ridge(
    dailyret: pd.DataFrame,
   firmchar: pd.DataFrame,
    window: int,
    z_list: list,
    jkp_factors: pd.DataFrame=None,
    linking_table: pd.DataFrame=None):
    """
    Computes portfolio returns using a rolling window strategy with Ridge Regression,
    for each shrinkage parameter in z_list.

    Args:
        dailyret (pd.DataFrame): DataFrame with daily returns and 'date' column.
        firmchar (pd.DataFrame): DataFrame with monthly characteristics and 'date' column.
        window (int): Number of months for training window.
        z_list (list): List of shrinkage (regularization) parameters for Ridge.

    Returns:
        list: A list of tuples (z, portfolio_returns_df, first_eval_date).
    """
    unique_dates = sorted(firmchar['date'].unique())

    if window < 2:
        raise ValueError("Window size must be at least 2.")

    results = []

    # Loop over each shrinkage parameter
    for z in z_list:
        print(f"\n=== Running Ridge with z = {z} ===")
        portfolio_returns = []
        first_eval_date = None

        # Rolling‐window loop
        for i in tqdm(range(window, len(unique_dates) - 6)):
            train_dates = unique_dates[i - window:i]
            test_date   = unique_dates[i]
            min_train, max_train = train_dates[0], train_dates[-1]
            print(f"Max train date = {max_train}, Test date = {test_date}")

            # Slice and merge
            tr_ret, tr_char = slice_dataframes(dailyret, firmchar,
                                               min_date=min_train,
                                               max_date=max_train,
                                               is_training=True)
            train_merged = (
                merge_all_data_frames(tr_ret, tr_char, jkp_factors=None, linking_table=linking_table)
                .reset_index()
                .drop(columns=['index_y', 'index', 'index_x'])
            )

            # Drop stocks with all‐NaN factors
            factor_cols = tr_char.columns[2:]
            train_merged = train_merged.dropna(subset=factor_cols, how='all').reset_index(drop=True)

            # Preprocess
            train_proc, cs_means, cols_to_drop = process_dataframe(train_merged)

            # Train ridge with this z
            betas_ridge, cols_in_train = train_ridge_regr(train_proc, [z])

            # Build test set
            te_ret, te_char = slice_dataframes(dailyret, firmchar,
                                               min_date=max_train,
                                               max_date=test_date,
                                               is_training=False)
            test_merged = (
                merge_all_data_frames(te_ret, te_char, jkp_factors=None, linking_table=linking_table)
                .reset_index()
                .drop(columns=['index_y', 'index', 'index_x'])
            )
            test_proc = test_merged.drop(columns=cols_to_drop)
            test_proc = test_proc[train_proc.columns]

            # Fill missing with cross‐sectional means
            feature_cols = [c for c in test_proc.columns
                            if c not in ['date','gvkey','DlyRet','PERMCO','PERMNO','SICCD','NAICS']]
            for c in feature_cols:
                if c in cs_means:
                    test_proc[c] = test_proc[c].fillna(cs_means[c])

            # Predict
            X_test = test_proc[cols_in_train]
            preds  = predict_ridge_regr(betas_ridge, X_test).to_numpy().ravel()

            preds_df = test_proc[['date','PERMCO','PERMNO']].copy()
            preds_df['prediction'] = preds
            weights  = normalize_weights(preds_df)

            test_proc['weight'] = weights.prediction
            test_proc['weighted_return'] = test_proc['weight'] * test_proc['DlyRet']

            daily_port_returns = (
                test_proc.groupby('date')['weighted_return']
                .sum()
                .reset_index()
            )

            portfolio_returns.append(daily_port_returns)
            gc.collect()

            if first_eval_date is None:
                first_eval_date = test_date

        # Combine and tag with z
        port_df = pd.concat(portfolio_returns, ignore_index=True)
        port_df['date'] = pd.to_datetime(port_df['date'])
        port_df['z']    = z

        results.append((z, port_df, first_eval_date))

    return results
    
    
