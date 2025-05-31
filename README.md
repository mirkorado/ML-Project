## Getting started
# Step 1: Make sure working directory is complete
Clone the repository and make sure your directory contains the following struture:
 - Root directory contains all .ipynb and .py files
 - Root directory contains folders 'Predictors' and 'Targets'
 - Predictors directory contains the following files: 'CompFirmCharac.csv' containing the data for firm characteristics and 'jkp.csv' containing the jkp factor returns
 - Targets directory contains: 'daily_crsp.csv' containing the daily returns to stocks
# Step 2: run data sanitization 
In a terminal and within a Python virtual environment containing the required packages (pandas) run the 'sanitizing.py' script which reads the above csv files, sanitizes the data and stores the sanitized data into new csv files. 
This script may take a few minutes to run. 

