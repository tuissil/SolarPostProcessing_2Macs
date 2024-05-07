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
    #torch.manual_seed(1234)  # default 42 # for riproducibility

    if trial is None:
        # keep last num_to_keep runs
        num_to_keep = 10
        keep_recent_files(f"{path}/runs", num_to_keep)  # not working
        writer = SummaryWriter()
    else:
        date_time = datetime.now().strftime("%d%m%y%H%M%S")
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