"""
Title: utils for QRNN from NWP
Created on May 07 2024
@email: lorenzo.tuissi@stiima.cnr.it
@author: Lorenzo Tuissi
"""


import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from torch.utils.data import Dataset
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Parameter
from torch.utils.data import DataLoader
import os
import glob
import time
import shutil
import scipy
import seaborn as sns

from copy import deepcopy
import itertools

import optuna
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import pickle
from torch import save
from datetime import datetime, timedelta
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from NWP_QR_data_import import split_data
from pytorch_lightning import seed_everything
from tools.cp_tools import compute_cqr, compute_pid, compute_cp
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class InvalidNumCaliSamples(Exception):
    pass


def truth_table(methods, step_wise_cp_options, num_cali_samples_true, num_cali_samples_false, p_gains, optimize_pid, test_months, T_burnin_arr):
    combinations = []

    for method in methods:
        for step_wise in step_wise_cp_options:
            num_cali_samples = num_cali_samples_true if step_wise else num_cali_samples_false
            for num_cali in num_cali_samples:
                if method == 'pid':
                    for p_gain in p_gains:
                        if optimize_pid:
                            # Add a row for each test month
                            for month in test_months:
                                for T_b in T_burnin_arr:
                                    if T_b >= num_cali:
                                        combinations.append([method, step_wise, num_cali, p_gain, month, num_cali])
                                    else:
                                        combinations.append([method, step_wise, num_cali, p_gain, month, T_b])
                        else:
                            for T_b in T_burnin_arr:
                                if T_b >= num_cali:
                                    combinations.append([method, step_wise, num_cali, p_gain, None, num_cali])
                                else:
                                    combinations.append([method, step_wise, num_cali, p_gain, None, T_b])

                elif (method=='cqr' or method=='cp') and optimize_pid:
                    for month in test_months:
                        combinations.append([method, step_wise, num_cali, None, month, None])
                else:
                    combinations.append([method, step_wise, num_cali, None, None, None])

    # Create the DataFrame
    df = pd.DataFrame(combinations, columns=['method', 'step_wise_cp', 'num_cali_samples', 'p_gain', 'test_set_number', 'T_burnin'])

    return df

def create_folder(path):
    try:
        os.makedirs(f"{path}/execution")
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(f"{path}/hyperparam_optim")
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(f"{path}/runs")
    except FileExistsError:
        # directory already exists
        pass

    files = glob.glob(f'{path}/execution/*')
    for f in files:
        os.remove(f)
    return

def train_hyperoptim(settings_optimization):
    va_te_date_split = datetime.strftime(datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S") + timedelta(days=-settings_optimization["num_cali_samples"]),"%Y-%m-%d %H:%M:%S")

    task_params = {}  # dict where to save all the optimal params for each station

    # read NWP data
    data_tasks = split_data(task=settings_optimization["task_names"],
                      yr_tr=settings_optimization["yr_tr"],
                      yr_va=settings_optimization["yr_va"],
                      yr_te=settings_optimization["yr_te"],
                      va_te_date_split=va_te_date_split,
                      hours_range=range(9, 17+1), # default range(9, 17+1)
                      scale_data=settings_optimization["scale_data"],
                      shuffle_train=settings_optimization["shuffle_train"],
                      optimize_pid=settings_optimization["optimize_pid"],
                      test_set_number=settings_optimization["test_set_number"],
                      num_cali_days=settings_optimization["num_cali_samples"],    # valid just if optimize pid
                      )


    for task_name in settings_optimization["task_names"]:

        print(f"Elaborating station {task_name}")
        # adjust path to save results in the correct folder

        for n_ens in range(settings_optimization['n_ensembles']):
            path = f"{settings_optimization['main_path']}/experiments/{settings_optimization['num_cali_samples']}/{task_name}/{n_ens}"  #os.path.join(main_path, "experiments", task_name)

           # create apposite folder to save results inside task_name folder
            create_folder(path)

        # select data related to the current station
        data = data_tasks[task_name]

        settings = {
            "input_size": data.x_train.shape[1],
            "hidden_size": 2 ** settings_optimization["exp_hidden_size"],  # default 2**3
            "out_size": len(settings_optimization["quantiles"]),
            "n_hidden_layers": settings_optimization["num_hidden_layers"],  # default 1
            "linear_map": False
        }

        # Hyperparams
        train_dataloader = DataLoader(DataloaderDataset(data.x_train.astype('float32'), data.y_train.astype('float32')), batch_size=settings_optimization["batch_size"], shuffle=settings_optimization["shuffle_dataloader"])
        vali_dataloader = DataLoader(DataloaderDataset(data.x_vali.astype('float32'), data.y_vali.astype('float32')), batch_size=settings_optimization["batch_size"])
        test_dataloader = DataLoader(DataloaderDataset(data.x_test.astype('float32'), data.y_test.astype('float32')), batch_size=len(data.x_test))


        # hyperoptim just on the last of the ensemble members

        def objective(trial):
            torch.manual_seed(settings_optimization["initial_seed"])
            exp_learning_rate = trial.suggest_int("exp_learning_rate", -6, -3)  # default -5, -1
            exp_hidden_size = trial.suggest_int("exp_hidden_size", 1, 5, step=1)  # default: 1, 5
            num_layers = trial.suggest_int("num_layers", 1, 5, step=1)  # default: 1, 5

            settings_hyperoptim = {
                "input_size": settings["input_size"],
                "hidden_size": 2**exp_hidden_size,
                "out_size": settings["out_size"],
                "n_hidden_layers": num_layers,
                "linear_map": settings["linear_map"]
            }

            print(f"learning rate: 10^{exp_learning_rate}, hidden_dim: {2 ** exp_hidden_size}, num_layers: {num_layers}")

            net = NWP_net(settings=settings_hyperoptim)

            optimizer = optim.Adam(net.parameters(), lr=10**exp_learning_rate, weight_decay=settings_optimization["weight_decay_optimizer"])
            scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9, verbose=True)  # default gamma = 0.95

            vali_error = train_model_dataloader(
                net=net,
                criterion=settings_optimization["criterion"],
                optimizer=optimizer,
                scheduler=scheduler,
                decay_steps=settings_optimization["decay_steps"],  # used only if scheduler is present
                num_epochs=settings_optimization["num_epochs_hyperoptim"],
                train_data=train_dataloader,
                vali_data=vali_dataloader,
                test_data=test_dataloader,
                patience=settings_optimization["patience_hyperoptim"],
                path=path,
                Newton_method=settings_optimization["Newton_method"],  # set to True if you want to use a Newton method
                error_precision=settings_optimization["error_precision"],  # number of decimals of the error
                trial=trial,
                pruner_epochs=settings_optimization["pruner_epochs"],
                pruner_max=settings_optimization["pruner_max"],
                pruner_sensitivity=settings_optimization["pruner_sensitivity"],
            )
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return vali_error


        if settings_optimization["hyper_param_opt"]:

            study = optuna.create_study(direction="minimize")

            # initial guess
            study.enqueue_trial({"exp_learning_rate": exp_learning_rate, "exp_hidden_size": exp_hidden_size, "num_layers": settings["n_hidden_layers"]})  # initial guess

            study.optimize(objective, n_trials=settings_optimization["n_trials"])

            trials = study.trials_dataframe()
            pruned_trials = trials[trials["state"] == "PRUNED"]
            complete_trials = trials[trials["state"] == "COMPLETE"]

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            trial = study.best_trial
            print(f"Best trial:{trial.number}")

            print("  Value: ", trial.value)

            # print best hyperparams
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            exp_learning_rate = trial.params["exp_learning_rate"]
            exp_hidden_size = trial.params["exp_hidden_size"]
            num_layers = trial.params["num_layers"]

            settings_hyperoptim_final = {
                "input_size": settings["input_size"],
                "hidden_size": 2 ** exp_hidden_size,
                "out_size": settings["out_size"],
                "n_hidden_layers": num_layers,
                "linear_map": settings["linear_map"]
            }

            net = NWP_net(settings=settings_hyperoptim_final)

            optimizer = optim.Adam(net.parameters(), lr=10**exp_learning_rate, weight_decay=settings_optimization["weight_decay_optimizer"])

            settings = settings_hyperoptim_final


        # TRAIN WITHOUT OPTUNA
        optim_param = {}
        torch.manual_seed(settings_optimization["initial_seed"])

        if settings_optimization["train_model_start"]:
            for n_ens in range(settings_optimization["n_ensembles"]):

                net = NWP_net(settings=settings)

                if settings_optimization["Newton_method"]:
                    optimizer = optim.LBFGS(net.parameters(), lr=1e-4, tolerance_grad=1e-3, tolerance_change=1e-5,
                                            line_search_fn='strong_wolfe')
                else:
                    optimizer = optim.Adam(net.parameters(), lr=10**settings_optimization["exp_learning_rate"], weight_decay=settings_optimization["weight_decay_optimizer"])

                scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9,
                                                       verbose=True)  # gamma = 1 to deactivate the scheduler

                optim_param[n_ens] = train_model_dataloader(
                    net=net,
                    criterion=settings_optimization["criterion"],
                    optimizer=optimizer,
                    scheduler=scheduler,
                    decay_steps=settings_optimization["decay_steps"],  # used only if scheduler is present
                    num_epochs=settings_optimization["num_epochs"],
                    train_data=train_dataloader,
                    vali_data=vali_dataloader,
                    test_data=test_dataloader,
                    patience=settings_optimization["patience"],
                    path=f"{settings_optimization['main_path']}/experiments/{settings_optimization['num_cali_samples']}/{task_name}/{n_ens}",
                    Newton_method=settings_optimization["Newton_method"],  # set to True if you want to use a Newton method
                    error_precision=settings_optimization["error_precision"],  # number of decimals of the error
                )

            data_tasks[task_name].task_params = optim_param

        data_tasks[task_name].test_dataloader = test_dataloader

    return settings, data_tasks

def compute_metrics(data_tasks, settings_optimization, additional_settings, settings_cp, decimals_quantile_score, scale_back_data=True, optimize_pid=False, test_month=0):
    quantile_score_table = pd.DataFrame()
    quantile_score_table_conformal = pd.DataFrame()

    for task_name in settings_optimization["task_names"]:

        print(f"Results related to {task_name}")
        data = data_tasks[task_name]
        path = os.path.join(settings_optimization['main_path'], "experiments", str(settings_optimization['num_cali_samples']), task_name)

        for i in range(settings_optimization['n_ensembles']):
            net = NWP_net(settings=additional_settings)
            if settings_optimization["train_model_start"]:
                net.load_state_dict(data_tasks[task_name].task_params[i])
            else:
                net.load_state_dict(torch.load(f"{path}/{i}/last_optim_model"))

            # compute predictions
            for X, y in data_tasks[task_name].test_dataloader:
                    X_test, y_test = X, y

            # compute prediction of the ensemble model
            if i == 0:
                pred = net(X_test).detach().numpy()
            else:
                pred = pred + net(X_test).detach().numpy()

        pred = pred / settings_optimization['n_ensembles']

        if scale_back_data:
            y_test = data.scaler_y.inverse_transform(y_test.reshape(-1, 1))
            pred = data.scaler_y.inverse_transform(pred)
        else:
            y_test = np.array(y_test).reshape(-1, 1)  # to correct predictions quantiles

        pred_sorted = np.sort(pred)  # contains quantiles prediction

        if optimize_pid and test_month==0:
            samples_to_consider = settings_optimization["num_cali_samples"]*settings_optimization['pred_horiz']
            y_test = y_test[-samples_to_consider:]
            pred_sorted = pred_sorted[-samples_to_consider:]
            days_to_consider_quantile_score = 30  # days to exclude from the beginning
            samples_to_consider_quantile_score = days_to_consider_quantile_score*settings_optimization['pred_horiz']
            settings_cp["num_cali_samples"]=settings_optimization["days_cali"] # change to execute the conformance, and restore it after
        elif optimize_pid and test_month>0:
            samples_to_consider = len(data_tasks[task_name].data_te) # consider all the samples
            samples_to_consider_quantile_score = 30*settings_optimization['pred_horiz']  # for a fair comparison of the quantile score, we consider as test set just the year 2020
        else:
            samples_to_consider = len(data_tasks[task_name].data_te) # consider all the samples
            samples_to_consider_quantile_score = len(data_tasks[task_name].data_te.index[data_tasks[task_name].data_te.index.year == data_tasks[task_name].year_te[1]])  # for a fair comparison of the quantile score, we consider as test set just the year 2020

        # quantile score computation before calibration
        quantile_score_table[task_name] = [np.round(quantile_score(
            y=y_test.flatten()[-samples_to_consider_quantile_score:],
            f=pred_sorted[-samples_to_consider_quantile_score:],
            tau=settings_optimization['quantiles']
        ), decimals_quantile_score)
        ]

        # create dataframe with computed predictions
        # pred_quantiles_test contain true measures and quantile predictions
        # data_reli contains data for plotting the reliability plot
        data_tasks[task_name].pred_quantiles_test, data_tasks[task_name].data_reli_test = reliability_plot(
            quantiles=settings_optimization['quantiles'],
            y=y_test[-samples_to_consider:],
            pred_sorted=pred_sorted[-samples_to_consider:],
            task_name=task_name,
            plot_graph=False
        )


        # Save data for conformal predictions
        if optimize_pid:
            x_conformal_scaled = data_tasks[task_name].x_test[:samples_to_consider]
            index_conformal_df = data_tasks[task_name].data_te.index[:samples_to_consider]
        else:
            x_conformal_scaled = data_tasks[task_name].x_test
            index_conformal_df=data_tasks[task_name].data_te.index[-samples_to_consider:]

        pred_conformal_scaled = net(torch.Tensor(x_conformal_scaled)).detach()
        if scale_back_data:
            pred_conformal = np.sort(data_tasks[task_name].scaler_y.inverse_transform(pred_conformal_scaled))[-samples_to_consider:]
            y_test = data_tasks[task_name].scaler_y.inverse_transform(data_tasks[task_name].y_test.reshape(-1, 1))[-samples_to_consider:]
        else:
            pred_conformal = np.sort(pred_conformal_scaled)[-samples_to_consider:]
            y_test = data_tasks[task_name].y_test.reshape(-1, 1)[-samples_to_consider:]

        data_tasks[task_name].pred_conformal_df = pd.DataFrame(
            index=index_conformal_df,
            columns=["y"] + [quantile for quantile in settings_optimization["quantiles"]],
            data=np.concatenate((y_test, pred_conformal), axis=1))

        # CONFORMANCE ANALYSIS
        results_cp = {}
        df = data_tasks[task_name].pred_conformal_df

        # initialize lr_hist
        lr_hist = None

        # df=pred_conformal_df_scaled
        if settings_cp["cp_method"] == 'cp':
            results_cp[task_name] = compute_cp([df], settings=settings_cp)
        elif settings_cp["cp_method"] == 'cqr':
            results_cp[task_name] = compute_cqr([df], settings=settings_cp)
        elif settings_cp["cp_method"] == 'pid':
            # returns also the kp history
            results_cp[task_name], lr_hist = compute_pid([df], settings=settings_cp, lr=settings_cp['pid_lr'], KI=settings_cp['pid_KI'])

        if optimize_pid and test_month==0:
            settings_cp["num_cali_samples"] = settings_optimization["num_cali_samples"]

        data_tasks[task_name].lr_hist = lr_hist

        # compute quantile score after conformal
        quantile_score_table_conformal[task_name] = [np.round(quantile_score(
            y=results_cp[task_name]['y'][-samples_to_consider_quantile_score:].to_numpy(),
            f=results_cp[task_name][settings_optimization["quantiles"]][-samples_to_consider_quantile_score:].to_numpy(),#results_cp[task_name][settings_optimization["quantiles"][-samples_to_consider_quantile_score:].tolist()].to_numpy(),
            tau=settings_optimization["quantiles"]
            ), decimals_quantile_score)
        ]
        data_tasks[task_name].pred_after_conformal_df = results_cp[task_name]

        print('cp done')

    
    return quantile_score_table, quantile_score_table_conformal

# Function to calculate quantile score
# y = y_true
# f = quantile predictions
# tau = quantile array
def quantile_score(y, f, tau=np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])):
    pbs = np.zeros_like(f)
    for j in range(len(tau)):
        q = f[:, j]
        pbs[:, j] = np.where(y >= q, tau[j] * (y - q), (1 - tau[j]) * (q - y))  # adding a product for a constant does not change the results
    return np.mean(pbs)

class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for index, quantile in enumerate(self.quantiles):
            errors = target - preds[:, index]
            loss = torch.mean(torch.max((quantile - 1) * errors, quantile * errors))

            losses.append(loss)

        return torch.mean(torch.stack(losses))

HuberNorm_epsilon = 2e-3

class HuberNorm(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        loss = []
        global HuberNorm_epsilon
        for i, q in enumerate(self.quantiles):
            error = target - preds[:, i]
            error_abs = torch.abs(error)
            h = torch.where(error_abs > HuberNorm_epsilon, error_abs - HuberNorm_epsilon / 2, (error ** 2) / (2 * HuberNorm_epsilon))
            loss_term = torch.where(error > 0, q * h * error, (q-1) * h * error)
            loss.append(loss_term)
        L = torch.stack(loss)
        total_loss = torch.mean(L)
        return total_loss

class DataloaderDataset(Dataset):
    def __init__(self,
                 x_data: torch.tensor,
                 y_data: torch.tensor
                 ) -> None:

        super().__init__()
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, i):
        x = self.x_data[i]
        y = self.y_data[i]
        return x, y



class NWP_net(nn.Module):

    def __init__(self, settings):
        super(NWP_net, self).__init__()
        self.settings = settings

        self.input_layer = \
                nn.Linear(self.settings['input_size'], self.settings['hidden_size'])


        self.hidden_layer = \
            nn.Sequential(
                nn.Linear(self.settings['hidden_size'], self.settings['hidden_size']),
                nn.ReLU()
            )

        self.hidden_layers = \
            nn.ModuleList([self.hidden_layer for _ in range(self.settings['n_hidden_layers'])])

        self.output_layer = \
            nn.Sequential(
                nn.Linear(self.settings['hidden_size'], self.settings['out_size']),
                nn.ReLU()
            )

    def forward(self, x):
        if self.settings["linear_map"]:
            x_out = x
        else:
            x_out = self.input_layer(x)
            for hidden_layer in self.hidden_layers:
                x_out = hidden_layer(x_out)

        logit = self.output_layer(x_out)

        return logit



min_epoch_error = np.inf

def train_model_dataloader(net, criterion, optimizer, num_epochs, train_data, vali_data, test_data, path, patience, scheduler=None, decay_steps=100, trial=None, pruner_epochs=0, pruner_max=0, pruner_sensitivity=np.inf, perfs_on_test=False, Newton_method=False, error_precision=8, n_ens=0): #
    date_time = datetime.now().strftime("%d%m%y%H%M%S")
    if trial is None:
        # keep last num_to_keep runs
        num_to_keep = 10
        keep_recent_files(f"{path}/runs", num_to_keep)
        writer = SummaryWriter(f"{path}/runs/{date_time}")
    else:
        print(f"Start optimization: {date_time}")
        folder_path=f"{path}/trials_logs"
        if trial._trial_id == 0:
            keep_recent_files(folder_path, 0)

        writer = SummaryWriter(log_dir=f"{folder_path}/{date_time}-trial{trial._trial_id}")


    continue_optimize = 1
    counter_stop = -1
    pruner_counter_epochs = 0

    epoch_error_best = 0


    train_loss_epoch = 0
    vali_loss_epoch = 0
    test_loss_epoch = 0

    global min_epoch_error
    global HuberNorm_epsilon

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}\n")
        net.train()

        if continue_optimize == 1:
            train_loss_batch = 0.0

            for X, y in train_data:
                if Newton_method:
                    def closure():
                        optimizer.zero_grad()
                        output = net(X)
                        train_loss = criterion(output, y)
                        if train_loss.isnan():
                            print("Train error is NaN")
                            return
                        train_loss.backward()
                        return train_loss
                    optimizer.step(closure)
                    output = net(X)
                    train_loss = criterion(output, y)
                else:
                    optimizer.zero_grad()
                    output = net(X)
                    train_loss = criterion(output, y)
                    if train_loss.isnan():
                        print("Train error is NaN")
                        break
                    train_loss.backward()
                    optimizer.step()


                train_loss_batch = train_loss_batch + train_loss

            # train loss epoch
            train_loss_epoch = round((train_loss_batch/len(train_data)).item(), error_precision)


            net.eval()
            vali_loss_epoch = round(vali_model(net=net, vali_data=vali_data, criterion=criterion).item(), error_precision)

            test_loss_epoch = round(vali_model(net=net, vali_data=test_data, criterion=criterion).item(), error_precision)

        # Exponential decay 
        if scheduler!=None and epoch > 0 and epoch % (decay_steps) == 0:
            scheduler.step()


        writer.add_scalar("Loss/train", train_loss_epoch, epoch)
        writer.add_scalar("Loss/vali", vali_loss_epoch, epoch)
        writer.add_scalar("Loss/test", test_loss_epoch, epoch)

        print(f"Epoch {epoch} - train loss: {train_loss_epoch}, vali loss: {vali_loss_epoch}, test loss: {test_loss_epoch}")

        torch.save(net.state_dict(), f'{path}/execution/epoch{epoch}')


        # compute the performance on test set -> just for debug
        if perfs_on_test == False:
            epoch_error = vali_loss_epoch
        else:
            epoch_error = test_loss_epoch

        # update min_epoch_error during optuna trials
        if epoch_error < min_epoch_error:
            min_epoch_error = epoch_error
            print(f"New minimum: {min_epoch_error}")

        # early stopping algorithm
        if epoch > 0 and epoch_error_best <= epoch_error:  # if the error increases
            counter_stop = counter_stop + 1
        else:
            epoch_error_best = epoch_error
            epoch_error_best_vali = vali_loss_epoch
            epoch_error_best_train = train_loss_epoch
            epoch_error_best_test = test_loss_epoch
            optim_param = deepcopy(net.state_dict())
            optim_epoch = epoch
            counter_stop = 0
            print("Saving the model!")

            if trial is None:
                torch.save(net.state_dict(),
                           f"{path}/last_optim_model")
            else:
                torch.save(net.state_dict(),
                           f"{path}/hyperparam_optim/best_model_trial_{trial.number}")

        print(f"Counter patience: {counter_stop}")


        if counter_stop == patience:
            continue_optimize = 0
            print("Early stopping stops the optimization")


        # pruning strategies
        if pruner_epochs > 0 and vali_loss_epoch > min_epoch_error*pruner_sensitivity:  # > pruner_max
            pruner_counter_epochs = pruner_counter_epochs + 1
            if pruner_counter_epochs >= pruner_epochs:
                continue_optimize = 0
                print(f"Pruning the trial: validation error exceeds the upper bound for {pruner_epochs} epochs (upper bound = {min_epoch_error*pruner_sensitivity})")

    if trial is not None:
        trial.report(epoch_error, epoch)

        # save hyperoptim results in a csv
        hyperoptim_res_epoch = pd.DataFrame(trial.params, index=[0])
        error_res_DF = pd.DataFrame({
            "Validation loss": [epoch_error_best_vali],
            "Train loss": [epoch_error_best_train],
            "Test loss": [epoch_error_best_test]
        })
        hyperoptim_res_epoch = pd.concat([hyperoptim_res_epoch, error_res_DF], axis=1)
        if trial.number == 0:
            hyperoptim_res_epoch.to_csv(f"{path}/trials_logs/trials_params_list.csv", sep=';')
        else:
            trials_params_list = pd.read_csv(f"{path}/trials_logs/trials_params_list.csv", sep=';', index_col=0)
            trials_params_list = pd.concat([trials_params_list, hyperoptim_res_epoch], axis=0, ignore_index=True)
            trials_params_list.to_csv(f"{path}/trials_logs/trials_params_list.csv", sep=';')



    print(f"Best model found at epoch {optim_epoch}")

    writer.flush()
    writer.close()

    if trial is None:
        return optim_param
    else:
        return epoch_error_best

def vali_model(net, vali_data, criterion):
    # VALIDATION
    vali_loss_batch = 0.0

    loss_rel = []

    for X, y in vali_data:
        output = net(X)
        vali_loss = criterion(output, y)
        vali_loss_batch = vali_loss_batch + vali_loss

        loss_rel.append(quantile_score(y, output.detach()))


    return vali_loss_batch/len(vali_data)+1*np.mean(loss_rel)  # vali loss across the dataloader

def keep_recent_files(folder_path, num_to_keep=10):

    # adjust folder_path format
    folder_path = folder_path.replace("/", "\\")

    # Get a list of all items (files and directories) in the folder sorted by modification time
    all_items = glob.glob(os.path.join(folder_path, '*'))
    all_items.sort(key=os.path.getmtime, reverse=True)

    # Separate items into files and directories
    files_to_remove = [item for item in all_items if os.path.isfile(item)][num_to_keep:]
    dirs_to_remove = [item for item in all_items if os.path.isdir(item)][num_to_keep:]

    # Remove the specified number of recent files and directories, handling permission issues
    for item_path in files_to_remove + dirs_to_remove:
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except PermissionError:
            print(f"Permission error: Unable to remove '{item_path}'")

def reliability_plot(quantiles, y, pred_sorted, task_name, plot_graph=True, conformal_title=False):
    tmp_y = pd.DataFrame({"y": y.flatten()})
    tmp_quantiles = pd.DataFrame(columns=quantiles, data=pred_sorted)
    scores_res = pd.concat([tmp_y, tmp_quantiles], axis=1)
    del tmp_y, tmp_quantiles

    # reliability_plot
    # definition of consistency bands
    confidence_90 = 0.9
    zscore_90 = scipy.stats.norm.ppf((1 + confidence_90) / 2)
    confidence_50 = 0.5
    zscore_50 = scipy.stats.norm.ppf((1 + confidence_50) / 2)

    # Construct new DataFrame format for reliability
    data_reli = pd.DataFrame()
    for j, tau in enumerate(quantiles):
        qtau = pred_sorted[:, j]
        P1 = (y < qtau.reshape(-1, 1)).mean()

        # bands
        mean = tau
        std_dev = np.std(y < qtau.reshape(-1, 1))
        upper90 = min(mean + zscore_90 * std_dev / np.sqrt(y.shape[0]), 1)
        lower90 = max(mean - zscore_90 * std_dev / np.sqrt(y.shape[0]), 0)

        upper50 = min(mean + zscore_50 * std_dev / np.sqrt(y.shape[0]), 1)
        lower50 = max(mean - zscore_50 * std_dev / np.sqrt(y.shape[0]), 0)

        data_reli = data_reli._append(pd.DataFrame({"x": [tau], "std": [std_dev], "P1": [P1], "CI50_lower": [lower50], "CI50_upper": [upper50], "CI90_lower": [lower90], "CI90_upper": [upper90]}), ignore_index=True)  #, "method": method, "stn": stn.upper()

    # Plot Reliability
    if plot_graph:
        fig, ax = plt.subplots(figsize=(10, 6))
        #for method, group in data_band.groupby("method"):
        #    ax.plot(group["x"], group["value"], label=method)
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.fill_between(data_reli["x"], data_reli["CI50_lower"], data_reli["CI50_upper"], alpha=0.5, color="royalblue", label="CI50")
        ax.fill_between(data_reli["x"], data_reli["CI90_lower"], data_reli["CI90_upper"], alpha=0.5, color="cornflowerblue", label="CI90")
        ax.plot(data_reli["x"], data_reli["P1"], color="grey")
        ax.scatter(data_reli["x"], data_reli["P1"], color='black', s=10, label="PICP")
        ax.set_xlabel("Quantile level (tau)")
        ax.set_ylabel("Coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        plt.grid()
        if conformal_title:
            plt.title(f"Reliability plot {task_name} - Conformal analysis")
        else:
            plt.title(f"Reliability plot {task_name}")
        plt.show()
    return scores_res, data_reli

def plot_sharpness_plot(scores_res, task_name, plot_graph=True):
    int_width_98 = scores_res[.99].values - scores_res[.01].values
    int_width_90 = scores_res[.95].values - scores_res[.05].values
    int_width_80 = scores_res[.9].values - scores_res[.1].values

    PIAW_98 = np.mean(int_width_98)
    PIAW_90 = np.mean(int_width_90)
    PIAW_80 = np.mean(int_width_80)

    # the sns catplot plot will represent
    # np.quantile(int_width_90, [0, 0.25, 0.5, 0.75, 1])
    # The “whiskers” extend to points that lie within 1.5 IQRs of the lower and upper quartile, and then observations
    # that fall outside this range are displayed independently
    # IQR = np.quantile(int_width_90, 0.75) - np.quantile(int_width_90, 0.25)
    # highest lines = quartile 0.75 + 1.5 * IQR
    # lowest line = quartile 0.25 -1.5*IQR
    tmp_sharp = pd.DataFrame({
        "80": int_width_80,
        "90": int_width_90,
        "98": int_width_98
    })

    tmp_sharp = pd.melt(tmp_sharp, var_name="interval", value_name="width")

    if plot_graph:
        # Plotting
        p2 = sns.catplot(data=tmp_sharp, kind="box", x="interval", y="width", palette="colorblind")
        # col_wrap=4, alpha=0.7, linewidth=1.5, width=0.6)
        p2.set_xlabels("Nominal coverage rate [%]")
        p2.set_ylabels("Interval width [W/m^2]")
        p2.set(ylim=(0, 1400), yticks=[0, 400, 800, 1200])
        plt.grid()
        plt.title(f"Sharpness plot {task_name}")
        plt.show()

    return PIAW_98, PIAW_90, PIAW_80

def plot_predictions_quantiles_fill(task_name, quantiles, y, pred_sorted, conformal_analysis, plot_graphs):
    if plot_graphs:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(y, 'b', label="Actual", linewidth=1.5)  # Actual values in blue

        # Set up the color gradient from yellow to red for each quantile pair
        cmap = cm.plasma  # "plasma" colormap goes from purple to yellow and red
        norm = Normalize(vmin=0, vmax=1)  # Normalize for range 0 to 1

        n_intervals = len(quantiles) // 2  # Number of quantile intervals
        for i in range(n_intervals):
            lower_idx = i
            upper_idx = -(i + 1)  # Pairs quantiles from outer to inner (e.g., 5th with 95th, 25th with 75th)

            # Reverse the quantile position to ensure consistent coloring
            quantile_position = i / (n_intervals - 1) if n_intervals > 1 else 0
            color = cmap(norm(1 - quantile_position))  # Flip color mapping to match legend

            # Plot the shaded region between the lower and upper quantile pairs
            ax.fill_between(
                np.arange(len(y)),  # x-axis
                pred_sorted[:, lower_idx],  # lower bound of fill
                pred_sorted[:, upper_idx],  # upper bound of fill
                color=color,  # Color based on the gradient
                label=f"{quantiles[lower_idx] * 100:.1f} - {quantiles[upper_idx] * 100:.1f}",
                alpha=0.9
            )

            # Plot lines for each quantile (optional for clarity)
            ax.plot(pred_sorted[:, lower_idx], color='g', linewidth=0.1)
            ax.plot(pred_sorted[:, upper_idx], color='g', linewidth=0.1)

        ax.legend(loc="upper left", fontsize="small")

        # Set title based on conformal analysis status
        title = f"Predictions Quantiles {task_name}{' - Conformal Analysis' if conformal_analysis else ''}"
        ax.set_title(title)
        ax.set_xlabel("Time [hours]")
        ax.set_ylabel(r"GHI [W/m$^2$]")
        ax.set_ylim(0, 800)

        ax.grid(True)

        # Set X-axis to represent hours
        hours = np.tile(np.arange(9, 18, 1), len(y)//9)
        hour_ticks = hours[::2]
        ax.set_xticks(ticks=np.linspace(0, len(y) - 1, num=len(hours))[::2])
        ax.set_xticklabels(hour_ticks)

        # Add the color gradient legend spanning from 0 to 1
        #sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        #sm.set_array([])
        #cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2)
        #cbar.set_label('Coverage level', fontsize=14)
        #cbar.ax.tick_params(labelsize=14)
        #cbar.set_ticks(np.linspace(0, 1, 5))  # Set ticks at 0.0, 0.25, 0.5, 0.75, 1.0
        #cbar.set_ticklabels([f"{tick:.2f}" for tick in np.linspace(0, 1, 5)])

        plt.show()


def compute_num_cali_samples(step_wise_cp, num_cali_samples_proposed, target_alpha, pred_horiz):
    if step_wise_cp:
        min_num_cali_samples = int((target_alpha[0])**-1)
        try:
            if num_cali_samples_proposed < min_num_cali_samples: raise InvalidNumCaliSamples() # compute minimum number of samples
        except:
            print("Invalid num_cali_samples, the default num_cali_samples will be used")
            num_cali_samples = 122
        else:
            num_cali_samples = num_cali_samples_proposed
    else:
        min_num_cali_samples = int((target_alpha[0] ** -1) / pred_horiz)
        try:
            if num_cali_samples_proposed < min_num_cali_samples: raise InvalidNumCaliSamples()
        except:
            print("Invalid num_cali_samples, the default num_cali_samples will be used")
            num_cali_samples = 10
        else:
            num_cali_samples = num_cali_samples_proposed
    return int(num_cali_samples)