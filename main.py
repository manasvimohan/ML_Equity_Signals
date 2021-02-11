'''
-----------------------------------------------------------------------
Created by Manasvi Mohan Sharma on 11/01/21 (dd/mm/yy)
Project Name: ML_Equity_Signals | File Name: main.py
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
This is the main file.
-----------------------------------------------------------------------
'''

import custom_functions
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn import (tree, linear_model, neighbors, ensemble, metrics)

# If required, we can scale data, but it is not required here
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# import matplotlib.pyplot as plt
# print(plt.style.available)
# plt.style.use('fivethirtyeight')

################################################

# START ###### Importing Data #######
# Define Start and end date of the time series data to be downloaded
years = 10
today = datetime.today().strftime('%Y-%m-%d')
start_date_calulated = datetime.today() - timedelta(days=years*365)
start_date = start_date_calulated.strftime('%Y-%m-%d')
today = datetime.today() - timedelta(days=4)
today = today.strftime('%Y-%m-%d')

timeframe = '1d'

#### Saves OHLC for NSEI or NSEBANK
symbolname = "^NSEBANK"
# symbolname = "^NSEI"
print('Modelling for {} index time series for simplicity for {} year(s)'.format(symbolname, years))
new_or_old = input("Download new data (new) or use saved data (saved)? : ").lower()

if new_or_old == 'new':
    print('Start date set to {} and end date to {}'.format(start_date, today))
    print('Getting OHLC for {} from {} to {}'.format(symbolname, start_date, today))
    df_symbol = custom_functions.make_df_OHLC_symbol(symbolname,start_date,today)
    file_name = 'All_Exports/01_Downloaded_Data/' + symbolname + '_input.csv'
    df_symbol.to_csv(file_name)
elif new_or_old == 'saved':
    file_name = 'All_Exports/01_Downloaded_Data/' + symbolname + '_input.csv'
    df_symbol = pd.read_csv(file_name)
    print('Loading existing data: {}'.format(file_name))
else:
    print("Error, please check filename or check for valid input")

# Specifics for the time series
data = df_symbol.copy()
column_name = input("Model on open, high, low or close? : ").lower()
lookbacklen = 250
period = 66
# END ###### Importing Data #######

# START ###### Setting up requirements #######
data['date'] = pd.to_datetime(data.index)
data = data[['date',column_name]].drop_duplicates()
data['date_index'] = data['date']
data.set_index('date_index', inplace=True)

diff_type = input("Do you wish to model on perc differences (yes/no): ").lower()
if diff_type == 'yes':
    data[column_name] = data[column_name].pct_change(period)
    data = data.iloc[period:]

    data = custom_functions.make_time_x(data, column_name, lookbacklen)
    temp = data.copy()
    calc_variable_name = column_name + '_' + str(period) + '_period'
    temp[calc_variable_name] = temp[column_name]
    data = temp.dropna()

elif diff_type == 'no':
    data = custom_functions.make_time_x(data, column_name, lookbacklen)
    temp = data.copy()
    calc_variable_name = column_name + '_' + str(period) + '_period'
    temp[calc_variable_name] = temp[column_name].diff(periods=period)
    temp[calc_variable_name] = temp[calc_variable_name].shift(periods=-period)
    data = temp.dropna()
else:
    print("Invalid input")

quantile_limit = 0.95
recommended_movement = data[calc_variable_name].quantile(q=quantile_limit)

if quantile_limit >= 0.50 and quantile_limit < 0.70:
    frequency = 'very high'
elif quantile_limit >= 0.70 and quantile_limit < 0.80:
    frequency = 'high'
elif quantile_limit >= 0.80 and quantile_limit < 0.90:
    frequency = 'medium'
elif quantile_limit >= 0.90 and quantile_limit < 0.95:
    frequency = 'low'
elif quantile_limit >= 0.95:
    frequency = 'very low'
else:
    frequency = 'NA'
    print('Invalid quantile')
# Creating tags in case of classification approach is taken
movement_required = recommended_movement
data, choice_name=  custom_functions.tag_data_variable_level(data,calc_variable_name,movement_required)
# END ###### Setting up requirements #######

# START ###### Some Data Cleaning #######
total_variables = list(data.columns)
y_variables = [column_name, calc_variable_name] # Y variables
y_choice_variables = [choice_name]
x_variables = list\
        (
              set(total_variables)
              - set(y_variables)
              - set(y_choice_variables)
        )
choice_column_name = y_choice_variables[0]
data = data.dropna()
# END ###### Some Data Cleaning #######

# START ###### Converting x and y to numpy arrays #######
x = data[x_variables].values
dummies = pd.get_dummies(data[choice_column_name]) # Classification (ONE HOT ENCODING DUMMY METHOD)
decisions = dummies.columns
y = dummies.values
print('--------------------------------------------------------')
print('Count of categories (y)')
custom_functions.count_y(y, 0)
custom_functions.count_y(y, 1)
custom_functions.count_y(y, 2)
# END ###### Converting x and y to numpy arrays #######

# START ###### Scaling x #######
# # MINMAX
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
# # STANDARD
# scaler = StandardScaler().fit(x)
# x = scaler.transform(x)
# END ###### Scaling x #######

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=9)

print('--------------------------------------------------------')
print('Count of categories (y_train)')
custom_functions.count_y(y_train, 0)
custom_functions.count_y(y_train, 1)
custom_functions.count_y(y_train, 2)

# START ###### Tackling imbalance data after train test split (RECOMMENDED) #######
print('--------------------------------------------------------')
balancing_required = input("Do you wish to balance out the training data (yes/no) : ").lower()

if balancing_required == 'yes':
    choice = input("Do you wish to random undersample (u), random oversample (o) or SMOTE (SMOTE): ").lower()
    x_train, y_train, choice = custom_functions.sample_data(x_train, y_train, choice)
    balancing_done = balancing_required
    print('Using balanced data')
    print('Final count of categories (y_train) after sampling')
    custom_functions.count_y(y_train, 0)
    custom_functions.count_y(y_train, 1)
    custom_functions.count_y(y_train, 2)

elif balancing_required == 'no':
    choice = 'none'
    balancing_done = 'no'
    print('Using unbalanced data')
    print('Final count of categories (y_train)')
    custom_functions.count_y(y_train, 0)
    custom_functions.count_y(y_train, 1)
    custom_functions.count_y(y_train, 2)
else:
    print('Invalid balance requirement given')

#########################################################

##########################################################
# # Standard Scaler
# scaler = StandardScaler().fit(x_train)
# x_train = scaler.transform(x_train)
# scaler = StandardScaler().fit(x_test)
# x_test = scaler.transform(x_test)
##########################################################

### The following is used to convert one hot encoding to label data by using array index approach
y_scikit_train = np.where(y_train==1)[1]
y_scikit_test = np.where(y_test==1)[1]
print('--------------------------------------------------------')
print('Modelling on {}'.format(column_name))
# print('Tick Length: {}'.format(value))
print('X Variables with lookback of {}'.format(lookbacklen))
print('Difference calculation period {}'.format(period))
print('Artificial tagging on historical data to capture movement of +/- {} points'.format(movement_required))
print('Decisions: {}'.format(list(decisions)))

###### Tree
clf_DTC = tree.DecisionTreeClassifier()
clf_DTC = clf_DTC.fit(x_train, y_scikit_train)

clf_ETC = tree.ExtraTreeClassifier()
clf_ETC = clf_ETC.fit(x_train, y_scikit_train)

###### Neighbhors
clf_KNC = neighbors.KNeighborsClassifier()
clf_KNC = clf_KNC.fit(x_train, y_scikit_train)

###### linear Model
clf_RCCV = linear_model.RidgeClassifierCV()
clf_RCCV = clf_RCCV.fit(x_train, y_scikit_train)

###### Ensemble
clf_RFC = ensemble.RandomForestClassifier()
clf_RFC = clf_RFC.fit(x_train, y_scikit_train)

clf_ETC_ens = ensemble.ExtraTreesClassifier()
clf_ETC_ens = clf_ETC_ens.fit(x_train, y_scikit_train)

clf_ABC = ensemble.AdaBoostClassifier()
clf_ABC = clf_ABC.fit(x_train, y_scikit_train)

clf_GBC = ensemble.GradientBoostingClassifier()
clf_GBC = clf_GBC.fit(x_train, y_scikit_train)

###### Using ensemble properly
def ensembleit(model):
    my_ensemble = ensemble.BaggingClassifier(model, max_samples=0.8, max_features=1.0) # Max features should be in float
    my_ensemble = my_ensemble.fit(x_train, y_scikit_train)
    return my_ensemble

my_ensemble = ensembleit(clf_KNC)

# START ###### Model Evaluation #######
model_name = clf_ETC_ens

print('\n--------------------------------------------------------')
print('\nModel Used: {}\n'.format(model_name))
print('----------------------Performance-----------------------')
pred = model_name.predict(x_test)
y_compare = np.argmax(y_test, axis=1)

acs = metrics.accuracy_score(y_compare, pred)
cm = metrics.confusion_matrix(y_compare, pred)
cr = metrics.classification_report(y_compare, pred)
cr_dict = metrics.classification_report(y_compare, pred, output_dict=True)

# # Plot Confusion Matrix
# try:
#     fig, ax = plt.subplots(figsize=(5, 5))
#     plt.rcParams.update({'font.size': 14})
#     metrics.plot_confusion_matrix(model_name, x_test, y_compare,
#                                   display_labels=decisions,
#                                   cmap=plt.cm.Blues,
#                                   ax=ax)  # doctest: +SKIP
#     plt.grid(False)
#     plt.show()
# except:
#     print("Confusion Matrix could not be printed")

success_buy, success_sell = custom_functions.buy_sell_acc(cm)

print("Accuracy score: {}".format(acs))
print("\nConfusion Matrix: \n\n{}".format(cm))
print("\nClassification Report: \n\n{}".format(cr))
# END ###### Model Evaluation #######

# START ###### Saving the model #######
print('--------------------------------------------------------')
save_model = input("\nDo you wish to save model (Yes/No): " ).lower()
to_date = datetime.now()
today_long = str(to_date.strftime("%Y-%m-%d %H %M %S"))

base_model_log_location = 'All_Exports/02_Exported_Models/'
pkl_filename = base_model_log_location + \
               symbolname+'_'+ column_name + \
               '_lb_'+ str(lookbacklen) + '_pc_' + str(period) +'_'+ choice +'_'+today_long + ".pkl"

# These variables will be saved in model log file

docx_log_files_location = 'All_Exports/03_Model_Logs/Docx_Logs/'+\
                          symbolname+'_'+ column_name + \
                          '_lb_'+ str(lookbacklen) + '_pc_' + str(period) +'_'+ choice +'_'+today_long+'.docx'
cm_cr = [cm,cr_dict]

variable_list = [symbolname,column_name,
                 lookbacklen, period, movement_required,
                 acs,success_buy, success_sell,
                 pkl_filename, model_name, balancing_required.lower(), choice, frequency, timeframe, diff_type, docx_log_files_location]

if save_model =='yes':
    custom_functions.export_model(variable_list, pkl_filename, model_name)
    custom_functions.export_model_log(docx_log_files_location, variable_list, cm_cr)
    print('Model Saved at: {}'.format(pkl_filename))
elif save_model =='no':
    print('Model not saved')
else:
    print('Not a valid input')
# END ###### Saving the model #######