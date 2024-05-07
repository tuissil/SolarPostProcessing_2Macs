"""
Title: QRNN from NWP
Created on May 07 2024
@email: lorenzo.tuissi@stiima.cnr.it
@author: Lorenzo Tuissi
"""
import numpy as np
import torch
from NWP_QR_data_import import split_data
from NWP_learning_utils import vali_model, DataloaderDataset, train_model_dataloader, PinballLoss, NWP_net, quantile_score, HuberNorm
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
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import scipy


path = os.getcwd().replace("\\", "/")
# remember to create the folder execution within the folder

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

files = glob.glob(f'{path}/execution/*')
for f in files:
    os.remove(f)


if __name__ == "__main__":


    seed_everything(0)  # default 42
    writer = SummaryWriter("runs")

    print("Start")
    task_name = "bon"  # ["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"]
    yr_tr = [2017, 2018]
    yr_va = [2019]
    yr_te = [2020]
    scale_data = True
    shuffle_train = False
    data = split_data(task=task_name,
                      yr_tr=yr_tr,
                      yr_va=yr_va,
                      yr_te=yr_te,
                      scale_data=scale_data,
                      shuffle_train=shuffle_train)

    # Hyperparams
    batch_size = 16  # default 64
    shuffle_dataloader = True  # data is already shuffled in the train set
    train_dataloader = DataLoader(DataloaderDataset(data.x_train.astype('float32'), data.y_train.astype('float32')), batch_size=batch_size, shuffle=shuffle_dataloader)
    vali_dataloader = DataLoader(DataloaderDataset(data.x_vali.astype('float32'), data.y_vali.astype('float32')), batch_size=batch_size)
    test_dataloader = DataLoader(DataloaderDataset(data.x_test.astype('float32'), data.y_test.astype('float32')), batch_size=len(data.x_test))


    ## Params
    quantiles = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]) # 0.01, 0.05, 0.95, 0.99
    exp_hidden_size = 3  # default 3
    settings = {
        "input_size": data.x_train.shape[1],
        "hidden_size": 2**exp_hidden_size,  # default 2**3
        "out_size": len(quantiles),
        "n_hidden_layers": 1, # default 1
        "linear_map": False
    }

    exp_learning_rate = -4  # default -4
    learning_rate = 10 ** exp_learning_rate
    num_epochs = 200  # default 200 100
    weight_decay_optimizer = 1e-3  # default 1e-3, L2 penalty
    decay_steps = 100  # default = 100, for learning rate scheduler
    error_precision = 4  # default: 4, number of decimals in the error

    n_ensembles = 1  # for ensemble NN
    
    patience = 50  # default 50
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
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9, verbose=True)
        
        vali_error = train_model_dataloader(net=net,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            scheduler=scheduler, #######scheduler,
                                            decay_steps=decay_steps,  # used only if scheduler is present
                                            num_epochs=num_epochs_hyperoptim,
                                            train_data=train_dataloader,
                                            vali_data=vali_dataloader,
                                            test_data=test_dataloader,
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
    train_model_start = False

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

            optim_param[i] = train_model_dataloader(net=net,
                                                 criterion=criterion,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler, #######scheduler,
                                                 decay_steps=decay_steps,  # used only if scheduler is present
                                                 num_epochs=num_epochs,
                                                 train_data=train_dataloader,
                                                 vali_data=vali_dataloader,
                                                 test_data=test_dataloader,
                                                 patience=patience,
                                                 path=path,
                                                 Newton_method=Newton_method,  # set to True if you want to use a Newton method
                                                 error_precision=error_precision,  # number of decimals of the error
                                                 )
            # optim_param according to the validation set
            torch.save(net.state_dict(), f"{path}/last_optim_model_his")


    # RESULTS

    print("Optimization results")
    scale_back_data_for_plot = True


    for i in range(n_ensembles):
        if train_model_start:
            net.load_state_dict(optim_param[i])
        else:
            net = NWP_net(settings=settings)
            net.load_state_dict(torch.load(f"{path}/last_optim_model_his"))
            
        # compute predictions
        for X, y in test_dataloader:
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




    # plot predictions quantiles
    plt.figure()
    plt.plot(y, 'b')
    for i in range(pred_sorted.shape[1]):
        plt.plot(pred_sorted[:, i], linewidth=0.5, label=i)
    plt.legend()
    plt.grid()
    plt.show()

    # quantile score computation
    quantile_score_table = quantile_score(y=y.flatten(),
                                          f=pred_sorted,
                                          tau=quantiles)
    print(quantile_score_table)

    # create dataframe with computed predictions
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

        data_reli = data_reli._append(pd.DataFrame({"x": [tau], "std": [std_dev], "y": [P1], "CI50_lower": [lower50], "CI50_upper": [upper50], "CI90_lower": [lower90], "CI90_upper": [upper90]}), ignore_index=True)  #, "method": method, "stn": stn.upper()

    # Plot Reliability
    fig, ax = plt.subplots(figsize=(10, 6))
    #for method, group in data_band.groupby("method"):
    #    ax.plot(group["x"], group["value"], label=method)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.fill_between(data_reli["x"], data_reli["CI50_lower"], data_reli["CI50_upper"], alpha=0.5, color="royalblue", label="CI50")
    ax.fill_between(data_reli["x"], data_reli["CI90_lower"], data_reli["CI90_upper"], alpha=0.5, color="cornflowerblue", label="CI90")
    ax.plot(data_reli["x"], data_reli["y"], color="grey")
    ax.scatter(data_reli["x"], data_reli["y"], color='black', s=10)
    ax.set_xlabel("Quantile level (tau)")
    ax.set_ylabel("Coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.grid()
    plt.title("Reliability plot")
    plt.show()

    # sharpness diagram
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
        #col_wrap=4, alpha=0.7, linewidth=1.5, width=0.6)
    p2.set_xlabels("Nominal coverage rate [%]")
    p2.set_ylabels("Interval width [W/m^2]")
    p2.set(ylim=(0, 1000))#, yticks=[0, 400, 800])
    plt.grid()
    plt.show()



    # export data to csv (to use in R code)
    data_CQRA = pd.read_csv("C:\\Users\\tuissi\\Documents\\NWP\\2Macs_code\\SolarPostProcessing_pytorch\\data\\bon_CQRA_2MACS_2019_2020.txt", sep='\t', index_col=0,  date_format="%Y-%m-%d %H:%M:%S")
    data_CQRA_2020 = data_CQRA[data_CQRA.index.year.isin([2020])]
    data_CQRA_2020[[col for col in data_CQRA_2020 if 'QRNN8.' in col]] = pred_sorted
    data_CQRA_2020.to_csv("C:\\Users\\tuissi\\Documents\\NWP\\code\\QuantileFcstComb-main\\Data\\bon_CQRA_2MACS_2020.txt", sep='\t', date_format="%Y-%m-%d %H:%M:%S")