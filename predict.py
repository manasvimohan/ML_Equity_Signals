'''
-----------------------------------------------------------------------
Created by Manasvi Mohan Sharma on 01/02/21 (dd/mm/yy)
Project Name: ML_Equity_Signals | File Name: predict.py
IDE: PyCharm | Python Version: 3.8
-----------------------------------------------------------------------
                                       _ 
                                      (_)
 _ __ ___   __ _ _ __   __ _ _____   ___ 
| '_ ` _ \ / _` | '_ \ / _` / __\ \ / / |
| | | | | | (_| | | | | (_| \__ \\ V /| |
|_| |_| |_|\__,_|_| |_|\__,_|___/ \_/ |_|

GitHub:   https://github.com/manasvimohan
Linkedin: https://www.linkedin.com/in/manasvi-mohan-sharma-119375168/
Website:  https://www.manasvi.co.in/
-----------------------------------------------------------------------
Project Information:
This uses classification approach to create buy/sell/hold signal
for any investment instrument with time series as input data, and
uses SKlearn library to model the series and predict there after
based on saved model

About this file:
This script is used to predict prices based on input time series
-----------------------------------------------------------------------
'''

from custom_functions import give_prediction
import pandas as pd
import custom_functions
from datetime import datetime, timedelta

# symbolname = "^NSEBANK"
symbolname = "^NSEI"
input_data_location = 'All_Inputs/'
log_file_location = 'All_Exports/03_Model_Logs/model_log.csv'

years = 2
today = datetime.today().strftime('%Y-%m-%d')
start_date_calulated = datetime.today() - timedelta(days=years*365)
start_date = start_date_calulated.strftime('%Y-%m-%d')
# today = datetime.today()- timedelta(days=1)

new_or_old = input("Download new data (new) or use saved data (saved)? : ").lower()

if new_or_old == 'new':
    df_symbol = custom_functions.make_df_OHLC_symbol(symbolname,start_date,today)
    input_file = input_data_location + symbolname + '_latest.csv'
    df_symbol.to_csv(input_file)
elif new_or_old == 'saved':
    input_file = input_data_location + symbolname + '_latest.csv'
    print('Loading existing data: {}'.format(input_file))
else:
    print("Error, please check filename or check for valid input")

log_file_df = pd.read_csv(log_file_location)
log_file_df = log_file_df.loc[log_file_df.symbol == symbolname]
log_file_df = log_file_df.loc[log_file_df.use_model == 'yes']
log_file_df = log_file_df.sort_values(by = 'timeframe').reset_index(drop=True)

m = log_file_df.shape[0]

# PROGRAMMING FOR LIVE FEED

###############
n=0
while n<m:
    choose_model = n
    print("---------------------------------------------")
    give_prediction(choose_model, input_file, log_file_df)
    n+=1