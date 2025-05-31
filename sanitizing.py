#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import os
import gc


def import_sanitize_daily_returns(target_path, nrows=None):
    """
    Import and sanitize the data frame of daily returns.

    Parameters:
     - target_path: the path to the CSV file containing the daily returns 
     - nrows: number of rows to load from the file. None if the whole dataset is to be loaded.
    """
    # Define only the necessary columns to import
    required_columns = ['PERMNO', 'PERMCO', 'DlyCalDt', 'SICCD', 'NAICS', 'DlyRet', 'sprtrn']

    # Load only the required columns
    daily_data = pd.read_csv(target_path, usecols=required_columns, nrows=nrows)

    # Drop rows with missing key data
    daily_data = daily_data.dropna(subset=['PERMNO', 'DlyCalDt', 'DlyRet'])

    # Convert dates
    daily_data['DlyCalDt'] = pd.to_datetime(daily_data['DlyCalDt'], errors='coerce')

    # Rename date column
    daily_data = daily_data.rename(columns={'DlyCalDt': 'date'})

    # Sort and deduplicate
    daily_data = daily_data.sort_values(by=['date', 'PERMCO', 'PERMNO'])
    daily_data = daily_data.drop_duplicates(subset=['PERMNO', 'date'])

    # Drop identifier columns 
    #drop_cols = ['CUSIP', 'HdrCUSIP', 'TradingSymbol', 'Ticker']
    #drop_cols = [col for col in drop_cols if col in daily_data.columns]
    #daily_data = daily_data.drop(columns=drop_cols)

    # Set multi-index
    daily_data = daily_data.set_index(['date', 'PERMCO', 'PERMNO'])

    return daily_data


def save_sanitized_csv(df, original_path, date_format='%Y-%m-%d'):
    """
    Save the sanitized DataFrame to a new CSV with '_sanitized' appended to the original filename.

    Parameters:
     - df: The sanitized DataFrame (with a datetime index).
     - original_path: The original CSV file path.
     - date_format: Format for datetime values in the output file.
    """
    # Get directory, filename, and extension
    base, ext = os.path.splitext(original_path)
    new_path = f"{base}_sanitized{ext}"

    # Reset index to save multi-index as columns
    df_to_save = df.reset_index()

    # Write to CSV with datetime formatting
    df_to_save.to_csv(new_path, index=False, date_format=date_format)

    print(f"Sanitized data saved to: {new_path}")


def import_sanitize_firm_charac(
    firm_charac_path, 
    nrows=None, 
    missing_threshold=0.5, 
    required_unique=500, 
    cutoff_date='1925-05-30'
):
    """
    Imports and sanitizes the dataset containing firm characteristics.

    Parameters:
     - firm_charac_path: path to CSV file containing the data
     - nrows: number of rows to load in memory (None if the whole dataset is to be loaded)
     - missing_threshold: if a column contains more than missing_threshold*100% missing values, it is dropped
     - required_unique: if a column has fewer than required_unique unique values, it is dropped
     - cutoff_date: data starts from this date (inclusive)
    """
    # Load data
    comp = pd.read_csv(firm_charac_path, nrows=nrows, low_memory=False)

    # Convert datadate to datetime
    comp['datadate'] = pd.to_datetime(comp['datadate'], errors='coerce')

    # Filter by date
    cutoff = pd.to_datetime(cutoff_date)
    comp = comp[comp['datadate'] >= cutoff]

    # Drop rows where gvkey or datadate is missing
    comp = comp.dropna(subset=['gvkey', 'datadate'])

    # Drop duplicate (gvkey, datadate) pairs
    comp = comp.drop_duplicates(subset=['gvkey', 'datadate'])

    # Rename datadate to date for consistency
    comp = comp.rename(columns={'datadate': 'date'})

    # Sort data frame and set multi-index
    comp = comp.sort_values(by=['date', 'gvkey'])
    comp = comp.set_index(['date', 'gvkey'])

    # Drop identifier columns
    identifiers = ['cusip', 'tic', 'conm', 'exchg', 'cik', 'costat', 'fic']
    comp = comp.drop(columns=[col for col in identifiers if col in comp.columns])

    # Drop columns with too many missing values
    valid_cols = comp.columns[comp.isna().mean() < missing_threshold]
    comp = comp[valid_cols]

    # Drop columns with too few unique values
    comp = comp.loc[:, comp.nunique(dropna=True) >= required_unique]

    return comp



def import_sanitize_jkp(jkp_path, nrows=None, cutoff_date='1925-05-30'):
    """
    Import and sanitize the factors. This function returns a data frame indexed by date
    containing only the returns of the factors. Columns location, frequency, weighting, 
    direction, n_stocks and n_stocks_min are dropped in the process because they are 
    uninformative. 
    Parameters: 
     - jkp_path: path to csv file containing the jkp factors. 
     - nrows: number of rows to load in memory. All of the data is loaded when nrows=None.
     - cutoff_date: the data starts from this date
    """
    jkp = pd.read_csv(jkp_path, nrows = 100000)
    
    jkp['date'] = pd.to_datetime(jkp['date'])    # Parse date column
    cutoff = pd.to_datetime(cutoff_date)
    jkp = jkp[jkp['date'] >= cutoff]        # Remove observations before 2000 as this is irrelevant for us

    # Pivot to wide format: one column per factor
    jkp_wide = jkp.pivot(index='date', columns='name', values='ret')

    return jkp_wide


def import_sanitize_linking_table(link_table_path):
    ccm = pd.read_csv(link_table_path)

    # Convert start date
    ccm['LINKDT'] = pd.to_datetime(ccm['LINKDT'])

    # Replace 'E' with a placeholder date, then convert and fill missing values
    ccm['LINKENDDT'] = ccm['LINKENDDT'].replace('E', '2099-12-31')
    ccm['LINKENDDT'] = pd.to_datetime(ccm['LINKENDDT'], errors='coerce')
    ccm['LINKENDDT'] = ccm['LINKENDDT'].fillna(pd.to_datetime('2099-12-31'))

    # Rename columns for merge compatibility
    ccm.rename(columns={'GVKEY': 'gvkey', 'LPERMNO': 'PERMNO', 'LPERMCO': 'PERMCO'}, inplace=True)

    ccm = ccm.set_index('gvkey')

    return ccm

def main():
    # Define paths and file names
    target_path = 'Targets/daily_crsp.csv'

    firm_charac_path = 'Predictors/CompFirmCharac.csv'
    jkp_path = 'Predictors/jkp.csv'
    # earnings_path = 'Predictors/earnings_calls.parquet'
    # mda_path = 'Predictors/mda_text.parquet'

    link_table_path = 'linking_table.csv'

    # First date for which we have daily returns
    cutoff_date = '2000-01-03'

    daily_data = import_sanitize_daily_returns(target_path, nrows=None)
    save_sanitized_csv(daily_data, target_path)
    del daily_data  # Remove the reference to the object
    gc.collect()  # Force garbage collection

    firms = import_sanitize_firm_charac(firm_charac_path, cutoff_date=cutoff_date)
    save_sanitized_csv(firms, firm_charac_path)
    del firms  # Remove the reference to the object
    gc.collect()  # Force garbage collection

    jkp = import_sanitize_jkp(jkp_path, cutoff_date=cutoff_date)
    save_sanitized_csv(jkp, jkp_path)
    del jkp  # Remove the reference to the object
    gc.collect()  # Force garbage collection

    links = import_sanitize_linking_table(link_table_path)
    save_sanitized_csv(links, link_table_path)
    del links  # Remove the reference to the object
    gc.collect()  # Force garbage collection


if __name__ == "__main__":
    main()





