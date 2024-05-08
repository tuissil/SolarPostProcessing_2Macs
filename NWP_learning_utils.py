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


# Function to calculate quantile score
# y = y_true
# f = quantile predictions
# tau = quantile array
def quantile_score(y, f, tau):
    pbs = np.zeros_like(f)
    for j in range(len(tau)):
        q = f[:, j]
        pbs[:, j] = np.where(y >= q, tau[j] * (y - q), (1 - tau[j]) * (q - y))
    return round(np.mean(pbs), 1)

class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for index, quantile in enumerate(self.quantiles):
            errors = target - preds[:, index]
            loss = torch.mean(torch.max((quantile - 1) * errors, quantile * errors))
            
            # add loss for reliability plot
            # loss = loss + 0.01 * abs(torch.mean(((target < preds[:, index]).float()))-quantile)
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
                #nn.Tanh()
                nn.ReLU()
            )

        self.hidden_layers = \
            nn.ModuleList([self.hidden_layer for _ in range(self.settings['n_hidden_layers'])])

        self.output_layer = \
            nn.Sequential(
                nn.Linear(self.settings['hidden_size'], self.settings['out_size']),
                nn.ReLU()
            )

        # identify function for uncensored regression quantiles
        # self.final_activation_param = nn.Parameter(torch.zeros(self.settings['out_size']), requires_grad=True)




    def forward(self, x):
        if self.settings["linear_map"]:
            x_out = x
        else:
            x_out = self.input_layer(x)
            for hidden_layer in self.hidden_layers:
                x_out = hidden_layer(x_out)

        logit = self.output_layer(x_out)

        # if identify function for uncensored regression quantiles
        #logit = torch.maximum(self.final_activation_param, self.output_layer(x_out))

        return logit



min_epoch_error = np.inf

def train_model_dataloader(net, criterion, optimizer, num_epochs, train_data, vali_data, test_data, path, patience, scheduler=None, decay_steps=100, trial=None, pruner_epochs=0, pruner_max=0, pruner_sensitivity=np.inf, perfs_on_test=False, Newton_method=False, error_precision=8): #
    torch.manual_seed(1234)  # default 42 # for riproducibility
    date_time = datetime.now().strftime("%d%m%y%H%M%S")
    if trial is None:
        # keep last num_to_keep runs
        num_to_keep = 10
        keep_recent_files(f"{path}/runs", num_to_keep)
        writer = SummaryWriter(f"{path}/runs/{date_time}")
    else:
        print(f"Start optimization: {date_time}")
        # delete old trials
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
                        # MPC train loss
                        # _, X1 = X
                        # train_loss = loss_MPC(train_loss, net, X1)
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
                    # MPC train loss
                    # _, X1 = X
                    # train_loss = loss_MPC(train_loss, net, X1)
                    train_loss.backward()
                    optimizer.step()


                train_loss_batch = train_loss_batch + train_loss

            # train loss epoch
            train_loss_epoch = round((train_loss_batch/len(train_data)).item(), error_precision)
            #print(f"Split {split}, epoch {epoch} - train loss: {train_loss_epoch[split]}")


            net.eval()
            vali_loss_epoch = round(vali_model(net=net, vali_data=vali_data, criterion=criterion).item(), error_precision)

            test_loss_epoch = round(vali_model(net=net, vali_data=test_data, criterion=criterion).item(), error_precision)

        # Exponential decay 
        if scheduler!=None and epoch > 0 and epoch % (decay_steps) == 0:
            scheduler.step()
            # HuberNorm_epsilon = HuberNorm_epsilon * 1e-2


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
            optim_param = net.state_dict()
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

    for X, y in vali_data:
        output = net(X)
        vali_loss = criterion(output, y)
        vali_loss_batch = vali_loss_batch + vali_loss


    return vali_loss_batch/len(vali_data)  # vali loss across the dataloader





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

def plot_reliability_plot(quantiles, y, pred_sorted, task_name):
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
    plt.title(f"Reliability plot {task_name}")
    plt.show()
    return scores_res, data_reli

def plot_sharpness_plot(scores_res, task_name):
    int_width_98 = scores_res[.99].values - scores_res[.01].values
    int_width_90 = scores_res[.95].values - scores_res[.05].values
    int_width_80 = scores_res[.9].values - scores_res[.1].values

    # the sns catplot plot will represent
    # np.quantile(int_width_90, [0, 0.25, 0.5, 0.75, 1])
    # The “whiskers” extend to points that lie within 1.5 IQRs of the lower and upper quartile, and then observations
    # that fall outside this range are displayed independently
    # IQR = np.quantile(int_width_90, 0.75) - np.quantile(int_width_90, 0.25)
    # highest lines = quartile 0.75 + 1.5 * IQR
    tmp_sharp = pd.DataFrame({
        "80": int_width_80,
        "90": int_width_90,
        "98": int_width_98
    })

    tmp_sharp = pd.melt(tmp_sharp, var_name="interval", value_name="width")

    # Plotting
    p2 = sns.catplot(data=tmp_sharp, kind="box", x="interval", y="width", palette="colorblind")
    # col_wrap=4, alpha=0.7, linewidth=1.5, width=0.6)
    p2.set_xlabels("Nominal coverage rate [%]")
    p2.set_ylabels("Interval width [W/m^2]")
    p2.set(ylim=(0, 1000))  # , yticks=[0, 400, 800])
    plt.grid()
    plt.title(f"Sharpness plot {task_name}")
    plt.show()