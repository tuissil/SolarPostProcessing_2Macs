"""
Title: import data for NWP
Created on April 05 2024
@email: lorenzo.tuissi@stiima.cnr.it
@author: Lorenzo Tuissi
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Class Stn_data definition
class Stn_data:
    def __init__(self, name):
        self.name = name  # station name
        self.x_train = None  # array with input train dataset
        self.y_train = None  # array with output train dataset
        self.x_vali = None  # array with input validation dataset
        self.y_vali = None  # array with output validation dataset
        self.x_test = None  # array with input test dataset
        self.y_test = None  # array with output test dataset
        self.scaler_x = None  # sklearn.preprocessing.StandardScaler() used to scale the input
        self.scaler_y = None  # sklearn.preprocessing.StandardScaler() used to scale the output
        self.data_te = None # test dataset for conformance prediction
        #self.target_test_df = None  # target df for ex-post data analysis
        self.data_original_df = None  # original dataframe
        self.pred_quantiles_test = None  # resulting predicted quantiles
        self.data_reli_test = None  # data for plotting reliability plot
        self.pred_conformal_df = None # dataframe with prediction for conformance analysis
        self.task_params = None  # models parameters
        self.test_dataloader = None  # test dataloader for prediction computation
        self.pred_after_conformal_df = None  # predictions quantiles after conformance
        self.year_te = None  # test set year for the conformal analysis

# separate the dataset in input and output
def in_out_split(data):
    try:
        if data[[col for col in data.columns if 'E_' in col]].empty or data['observations'].empty:
            raise Exception("Empty DataFrame")
        inp = data[[col for col in data.columns if 'E_' in col]].values
        out = data['observations'].values
    except:
        print("Empty dataframe to split")
        inp = None
        out = None
    return inp, out


# Create dictionary with instances of class Stn_data
# Associate to each station the corresponding train, (validation), and test I/O dataset as array in scaled or unscaled
# form according to parameter scale_data
def split_data(task, yr_tr, yr_te, yr_va, va_te_date_split, hours_range, scale_data=True, shuffle_train=False):
    dir =os.path.join(os.getcwd(), 'data_zenith100')  # default 'data'
    stns = task

    stns_list = {}
    for stn in stns:
        print(f"Processing station {stn}")
        stns_list[stn] = Stn_data(name=stn)
        data = pd.read_csv(os.path.join(dir, f"{stn}_obs_ECMWF_ens.csv"), sep=",", index_col=0, date_format="%Y-%m-%d %H:%M:%S")
        stns_list[stn].data_original_df = data
        
        # count different hours observations
        # data.index.hour.value_counts()
        
        # Train/test split
        if shuffle_train:
            data_tr = data[data.index.year.isin(yr_tr)].sample(frac=1)
        else:
            data_tr = data[data.index.year.isin(yr_tr)]
        #data_tr = data_tr[data_tr.index.hour.isin(hours_range)]  # if same hours range as in test set
        
        # validation data
        data_va = data[data.index.year.isin(yr_va)][:va_te_date_split]  # data[data.index.year.isin(yr_va)]
        #data_va = data_va[data_va.index.hour.isin(hours_range)]  # if same hours range as in test set

        # test data
        data_te = data[data.index.year.isin([yr_va[0], yr_te[0]])][va_te_date_split:]
        data_te = data_te[data_te.index.hour.isin(hours_range)] # modified for conformance  data[data.index.year.isin(yr_te)]
        
        #target_test_df = data_te[['observations']].copy()
        #target_test_df.rename(columns={'observations':stn}, inplace=True)
        if scale_data:
            scaler_output = MinMaxScaler().fit(
                data_tr["observations"].values.reshape(-1, 1))  # scaler for just the output
            scaler_input = MinMaxScaler().fit(
                data_tr[[col for col in data.columns if 'E_' in col]])  # scaler for just the input
            scaler_complete = MinMaxScaler().fit(data_tr)  # scaler including both output and input
            data_tr_values = scaler_complete.transform(data_tr)
            data_tr = pd.DataFrame(data_tr_values, columns=data_tr.columns, index=data_tr.index)
            data_te_values = scaler_complete.transform(data_te)
            data_te = pd.DataFrame(data_te_values, columns=data_te.columns, index=data_te.index)
            try:
                data_va_values = scaler_complete.transform(data_va)
                data_va = pd.DataFrame(data_va_values, columns=data_va.columns, index=data_va.index)
            except:
                print("Warning: empty dataset to scale")
                data_va = None
            stns_list[stn].scaler_x = scaler_input
            stns_list[stn].scaler_y = scaler_output

        # Input/output split
        stns_list[stn].x_train, stns_list[stn].y_train = in_out_split(data_tr)
        stns_list[stn].x_vali, stns_list[stn].y_vali = in_out_split(data_va)
        stns_list[stn].x_test, stns_list[stn].y_test = in_out_split(data_te)
        stns_list[stn].data_te = data_te
        stns_list[stn].year_te = yr_te

    return stns_list


    # stns_list is a dictionary of instances of class Stn_data:
    # the class has the following attributes
    # name: name of the station
    # train_data_in: array with input train dataset
    # train_data_out: array with output train dataset
    # vali_data_in: array with input validation dataset
    # vali_data_out: array with output validation dataset
    # test_data_in: array with input test dataset
    # test_data_out: array with output test dataset
    # For example, to return the test set input of the station 'bon': stns_list['bon'].test_data_in
    # Stations name ["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"]
    # Example: stn_data = split_data(task='bon', yr_tr=yr_tr, yr_te=yr_te, yr_va=yr_va)

