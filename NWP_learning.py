"""
Title: QRNN from NWP
Created on May 07 2024
@email: lorenzo.tuissi@stiima.cnr.it
@author: Lorenzo Tuissi
"""
import numpy as np
import torch
from NWP_QR_data_import import split_data
from NWP_learning_utils import vali_model, DataloaderDataset, train_model_dataloader, PinballLoss, NWP_net, quantile_score, HuberNorm, create_folder, reliability_plot, plot_sharpness_plot
from pytorch_lightning import seed_everything
import pickle
from torch import save
import pandas as pd
import optuna
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to remove info and warnings
import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


main_path = os.getcwd().replace("\\", "/")

if __name__ == "__main__":


    seed_everything(100)  # default 0 [bon = 0, fpk=42

    print("Start")
    task_names = ["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"] #["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"]
    yr_tr = [2017, 2018]
    yr_va = [2019]
    yr_te = [2020]
    scale_data = True
    shuffle_train = False

    ## Optim params
    quantiles = np.array(
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])  # 0.01, 0.05, 0.95, 0.99
    exp_hidden_size = 3  # default 3

    train_model_start = False  # retrain the model
    exp_learning_rate = -4  # default -4
    learning_rate = 10 ** exp_learning_rate
    num_epochs = 500  # default 500
    weight_decay_optimizer = 1e-3  # default 1e-3, L2 penalty
    decay_steps = 100  # default = 100, for learning rate scheduler
    error_precision = 4  # default: 4, number of decimals in the error

    batch_size = 64  # default 64
    shuffle_dataloader = True

    n_ensembles = 1  # for ensemble NN

    patience = 100  # default 100
    Newton_method = False  # default False, set to True if you want to adopt a Newton method, not valid for hyperparam optim

    criterion = PinballLoss(quantiles)  # HuberNorm(quantiles) #PinballLoss(quantiles)  # reduction='sum'

    # hyperoptim set
    hyper_param_opt = False
    num_epochs_hyperoptim = 150
    patience_hyperoptim = 50
    n_trials = 10
    pruner_epochs = 50
    pruner_max = 0.1  # if the computed error is above pruner_max, count patience
    pruner_sensitivity = 1.2  # multiply the min_epoch_error for this factor to increase the patience counter

    task_params = {}  # dict where to save all the optimal params for each station

    # read NWP data
    data_tasks = split_data(task=task_names,
                      yr_tr=yr_tr,
                      yr_va=yr_va,
                      yr_te=yr_te,
                      scale_data=scale_data,
                      shuffle_train=shuffle_train)

    test_dataloader = {}  # this dataloader will be used to display results

    for task_name in task_names:

        seed_everything(100)
        #torch.manual_seed(100)

        print(f"Elaborating station {task_name}")
        # adjust path to save results in the correct folder
        path = f"{main_path}/experiments/{task_name}"  #os.path.join(main_path, "experiments", task_name)

        #writer = SummaryWriter(os.path.join(path, "runs"))

        # create apposite folder to save results inside task_name folder
        create_folder(path)

        # select data related to the current station
        data = data_tasks[task_name]

        settings = {
            "input_size": data.x_train.shape[1],
            "hidden_size": 2 ** exp_hidden_size,  # default 2**3
            "out_size": len(quantiles),
            "n_hidden_layers": 1,  # default 1
            "linear_map": False
        }

        # Hyperparams
        train_dataloader = DataLoader(DataloaderDataset(data.x_train.astype('float32'), data.y_train.astype('float32')), batch_size=batch_size, shuffle=shuffle_dataloader)
        vali_dataloader = DataLoader(DataloaderDataset(data.x_vali.astype('float32'), data.y_vali.astype('float32')), batch_size=batch_size)
        test_dataloader[task_name] = DataLoader(DataloaderDataset(data.x_test.astype('float32'), data.y_test.astype('float32')), batch_size=len(data.x_test))

        def objective(trial):
            torch.manual_seed(42)
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

            optimizer = optim.Adam(net.parameters(), lr=10**exp_learning_rate, weight_decay=weight_decay_optimizer)
            scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9, verbose=True)  # default gamma = 0.95

            vali_error = train_model_dataloader(
                net=net,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler, #######scheduler,
                decay_steps=decay_steps,  # used only if scheduler is present
                num_epochs=num_epochs_hyperoptim,
                train_data=train_dataloader,
                vali_data=vali_dataloader,
                test_data=test_dataloader[task_name],
                patience=patience_hyperoptim,
                path=path,
                Newton_method=Newton_method,  # set to True if you want to use a Newton method
                error_precision=error_precision,  # number of decimals of the error
                trial=trial,
                pruner_epochs=pruner_epochs,
                pruner_max=pruner_max,
                pruner_sensitivity=pruner_sensitivity,
            )
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return vali_error


        if hyper_param_opt:

            study = optuna.create_study(direction="minimize")

            # initial guess
            study.enqueue_trial({"exp_learning_rate": exp_learning_rate, "exp_hidden_size": exp_hidden_size, "num_layers": settings["n_hidden_layers"]})  # initial guess

            study.optimize(objective, n_trials=n_trials)

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

            optimizer = optim.Adam(net.parameters(), lr=10**exp_learning_rate, weight_decay=weight_decay_optimizer)

            settings = settings_hyperoptim_final


        # TRAIN WITHOUT OPTUNA
        #net.load_state_dict(torch.load(f"{path}/last_optim_model_cogeneration"))
        #optim_param = train_model(net=net, criterion=criterion, optimizer=optimizer, patience=patience, num_epochs=num_epochs, training_data=training_data, vali_data=vali_data, enc_seq_len=enc_seq_len)

        optim_param = {}

        if train_model_start:
            for i in range(n_ensembles):

                net = NWP_net(settings=settings)

                if Newton_method:
                    optimizer = optim.LBFGS(net.parameters(), lr=1e-4, tolerance_grad=1e-3, tolerance_change=1e-5,
                                            line_search_fn='strong_wolfe')
                else:
                    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay_optimizer)

                scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9,
                                                       verbose=True)  # gamma = 1 to deactivate the scheduler

                optim_param[i] = train_model_dataloader(
                    net=net,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler, #######scheduler,
                    decay_steps=decay_steps,  # used only if scheduler is present
                    num_epochs=num_epochs,
                    train_data=train_dataloader,
                    vali_data=vali_dataloader,
                    test_data=test_dataloader[task_name],
                    patience=patience,
                    path=path,
                    Newton_method=Newton_method,  # set to True if you want to use a Newton method
                    error_precision=error_precision,  # number of decimals of the error
                )
            # optim_param according to the validation set
            #torch.save(net.state_dict(), f"{path}/last_optim_model_his")
            task_params[task_name] = optim_param


    # RESULTS
    print("Optimization results")
    scale_back_data_for_plot = True
    quantile_score_table = pd.DataFrame()

    for task_name in task_names:

        print(f"Results related to {task_name}")
        data = data_tasks[task_name]
        path = os.path.join(main_path, "experiments_github", task_name)

        for i in range(n_ensembles):
            if train_model_start:
                net.load_state_dict(task_params[task_name][i])
            else:
                net = NWP_net(settings=settings)
                net.load_state_dict(torch.load(f"{path}/last_optim_model"))
    
            # compute predictions
            for X, y in test_dataloader[task_name]:
                X_test, y_test = X, y
    
            # compute prediction of the ensemble model
            if i == 0:
                pred = net(X_test).detach().numpy()
            else:
                pred = pred + net(X_test).detach().numpy()
    
        pred = pred/n_ensembles
    
        if scale_back_data_for_plot:
            y = data.scaler_y.inverse_transform(y.reshape(-1,1))
            pred = data.scaler_y.inverse_transform(pred)
    
        pred_sorted = np.sort(pred)  # contains quantiles prediction
    
        '''
        # plot predictions quantiles
        plt.figure()
        plt.plot(y, 'b')
        for i in range(len(quantiles)):
            plt.plot(pred_sorted[:, i], linewidth=0.5, label=quantiles[i])
        plt.legend()
        plt.title(f"Predictions quantiles {task_name}")
        plt.grid()
        plt.show()
        '''
    
        # quantile score computation
        quantile_score_table[task_name] = [quantile_score(
            y=y.flatten(),
            f=pred_sorted,
            tau=quantiles
            )
        ]
       
    
        # create dataframe with computed predictions
        # scores_quantiles contain true measures and quantile predictions
        # data_reli contains data for plotting the reliability plot
        data_tasks[task_name].pred_quantiles_test, data_tasks[task_name].data_reli_test = reliability_plot(
            quantiles=quantiles, 
            y=y, 
            pred_sorted=pred_sorted,
            task_name=task_name,
            plot_graph=True
        )
    
        # sharpness diagram
        plot_sharpness_plot(data_tasks[task_name].pred_quantiles_test, task_name)

        # Save data for conformance predictions
        data_conformance = data_tasks[task_name].data_original_df["2019-10-01 00:00:00":]
        x_conformance_scaled = data_tasks[task_name].scaler_x.transform(data_conformance[[col for col in data_conformance.columns if 'E_' in col]])
        pred_conformance_scaled = net(torch.Tensor(x_conformance_scaled)).detach()
        pred_conformance = np.sort(data_tasks[task_name].scaler_y.inverse_transform(pred_conformance_scaled))

        data_tasks[task_name].pred_conformance_df = pd.DataFrame(
            index=data_conformance.index,
            columns=["y"]+['Q' + str(quantile) for quantile in quantiles],
            data=np.concatenate((data_conformance["observations"].values.reshape(-1,1), pred_conformance), axis=1))

        '''
        # if same predictions but scaled
        pred_conformance_df_scaled = pd.DataFrame(index=data_conformance.index,
                                                  columns=["y"]+['Q' + str(quantile) for quantile in quantiles],
                                                  data=np.concatenate((data_tasks[task_name].scaler_y.transform(data_conformance["observations"].values.reshape(-1,1)),
                                                           pred_conformance_scaled), axis=1))
        '''

        # export data to csv (to use in R code)
        '''
        data_CQRA = pd.read_csv("C:\\Users\\tuissi\\Documents\\NWP\\2Macs_code\\SolarPostProcessing_pytorch\\data\\bon_CQRA_2MACS_2019_2020.txt", sep='\t', index_col=0,  date_format="%Y-%m-%d %H:%M:%S")
        data_CQRA_2020 = data_CQRA[data_CQRA.index.year.isin([2020])]
        data_CQRA_2020[[col for col in data_CQRA_2020 if 'QRNN8.' in col]] = pred_sorted
        data_CQRA_2020.to_csv("C:\\Users\\tuissi\\Documents\\NWP\\code\\QuantileFcstComb-main\\Data\\bon_CQRA_2MACS_2020.txt", sep='\t', date_format="%Y-%m-%d %H:%M:%S")
        '''

    print(quantile_score_table)
    print("End execution")
