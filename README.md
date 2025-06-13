# Getting started

## Step 1: Make sure working directory is complete

Clone the repository and make sure your directory contains the following struture:
 - Root directory contains all .ipynb and .py files
 - Root directory contains the linking table (between PERMNO and gvkey used for merging datasets from different databases) 'linking_table.csv'
 - Root directory contains folders 'Predictors' and 'Targets'
 - Predictors directory contains the following files: 'CompFirmCharac.csv' containing the data for firm characteristics 
 - Targets directory contains: 'daily_crsp.csv' containing the daily returns to stocks


## Step 2: run data sanitization 

In a terminal and within a Python virtual environment containing the required packages (pandas) run the 'sanitizing.py' script which reads the above csv files, sanitizes the data and stores the sanitized data into new csv files. 

This script may take a few minutes to run and is memory intensive. It has been shown to work on a pc with 16GB of available RAM and 4GB of swap memory. 

## Step 3: run the notebook Build_Firm_Index.ipynb 

This notebook builds a column index for each stock at a given date based on its firm's monthly characteristics. The approach uses partial least squares (PLS) trained on monthly cumulative returns to build the column index, capturing the most variation possible. 

## Optional step: run the notebook Tune_DNN_Params.ipynb

This notebook's aim is to find appropriate parameters and the best loss function for the implemented deep neural network which optimize training speed and provide a coherent, diversified trading strategy. We experiment with the first 7 months of data (6 months for training and 1 month for testing) as this will be the rolling window used in the final strategy.

## Note:

For lack of access to GPUs, our model trains the DNN on the maximum amount of CPU cores available (16 in our case). Training the model is therefore time consuming. 
