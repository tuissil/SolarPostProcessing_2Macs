"""
Title: QRNN from NWP
Created on May 07 2024
@email: lorenzo.tuissi@stiima.cnr.it
@author: Lorenzo Tuissi
"""
# ------------------------------------------------------------
# configs
# ------------------------------------------------------------
# TODO:
# X definire split validation con test che comprende calibration subset a seconda della definizione nel setting
# definire grid per lanciare esperimenti sulle diverse configurazioni e salvare risultati pickle
# fare prove con diversi orizzonti passato
# X implementare calcolo score sui modelli conformal

# future investigazioni:
# autotuning pid (ste)
# fare prove con senza standardizzazione (valutare "detrending")



import numpy as np
import torch
from NWP_learning_utils import vali_model, DataloaderDataset, train_model_dataloader, PinballLoss, NWP_net, quantile_score, HuberNorm, create_folder, reliability_plot, plot_sharpness_plot, truth_table, train_hyperoptim, compute_metrics, plot_predictions_quantiles, compute_num_cali_samples
from pytorch_lightning import seed_everything
import pandas as pd


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to remove info and warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import pickle


main_path = os.getcwd().replace("\\", "/")






if __name__ == "__main__":

    print("Start")

    # CONFORMANCE ANALYSIS

    target_alpha = [0.02, 0.1, 0.2, 0.4, 0.6, 0.8]
    pred_horiz = 9  # default 9


    # create truth table for the proofs
    methods = ['cp', 'cqr', 'pid'] #['cp', 'cqr', 'pid']
    step_wise_cp_options = [True, False]
    num_cali_samples_true = [50, 122, 150]  # if step_wise true
    num_cali_samples_false = [15, 10, 5]  # if not step_wise
    truth_table = truth_table(methods, step_wise_cp_options, num_cali_samples_true, num_cali_samples_false)
    truth_table["data_tasks"] = np.NaN
    truth_table["quantile_score"] = np.NaN
    truth_table["quantile_score_conformal"] = np.NaN

    with open(f"{main_path}/num_cali_samples_observed", "rb") as fp:  # save list with already observed samples
        num_cali_samples_observed = pickle.load(fp)

    for i in range(truth_table.__len__()):
        cp_method = truth_table["method"][i] # 'cp', 'cqr', 'pid  # con pid usare step_wise_cp = True perchè non ho ancora implementato la versione non stepwise
        step_wise_cp = truth_table["step_wise_cp"][i]

        num_cali_samples = compute_num_cali_samples(
            step_wise_cp,
            truth_table["num_cali_samples"][i],
            target_alpha,
            pred_horiz
        )

        if num_cali_samples not in num_cali_samples_observed:
            train_model_start = True
            num_cali_samples_observed.append(num_cali_samples)
        else:
            train_model_start = False

        settings_cp = {
            'target_name': 'y',
            'num_cali_samples': num_cali_samples,  # number of past days of the test day to use for the conformance analysis
            'target_alpha': target_alpha,
            'pred_horiz': pred_horiz,  # number of hours to predict (it should not be changed)
            'num_ense': 1,
            'stepwise': step_wise_cp,
            'cp_method': cp_method,
            'pid_lr': 0.01,  # just if cp_method = pid
            'pid_KI': 10,  # just if cp_method = pid
        }

        print(f"Truth table row {i}, Method: {cp_method}, stepwise: {step_wise_cp}, num_cali_samples: {num_cali_samples}")

        #  Parameters setting for training and hyperoptimization
        quantiles = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        settings_optimization = {
            "task_names": ["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"],  #["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"]
            "yr_tr": [2017, 2018],
            "yr_va": [2019],
            "yr_te": [2020],
            "scale_data": True,
            "shuffle_train": False,
            ## Optim params
            "quantiles": quantiles,  # 0.01, 0.05, 0.95, 0.99
            "exp_hidden_size": 3,  # default 3
            "num_hidden_layers": 1,  # default 1
            "initial_seed": 0,  # default 0, same for both trainining and optimization of hyperparameters
            "main_path": main_path,

            "train_model_start": train_model_start,  # retrain the model
            "exp_learning_rate": -4,  # default -4
            "num_epochs": 500,  # default 500
            "weight_decay_optimizer": 1e-3,  # default 1e-3, L2 penalty
            "decay_steps": 100,  # default = 100, for learning rate scheduler
            "error_precision": 4,  # default: 4, number of decimals in the error
            "batch_size": 64,  # default 64
            "shuffle_dataloader": True,
            "n_ensembles": 1,  # for ensemble NN
            "patience": 100,  # default 100
            "Newton_method": False,  # default False, set to True if you want to adopt a Newton method, not valid for hyperparam optim
            "criterion": PinballLoss(quantiles),  # HuberNorm(quantiles) #PinballLoss(quantiles)  # reduction='sum'
            "num_cali_samples": num_cali_samples,  # default = 122, number of days subtracted from yr_va in the validation set

            # hyperoptim set
            "hyper_param_opt": False,
            "num_epochs_hyperoptim": 150,  # default = 150
            "patience_hyperoptim": 50,  # default = 50
            "n_trials": 10,  # default = 10
            "pruner_epochs": 50,  # default = 50
            "pruner_max": 0.1,  # default = 0.1, if the computed error is above pruner_max, count patience
            "pruner_sensitivity": 1.2,  # default = 1.2, multiply the min_epoch_error for this factor to increase the patience counter
        }

        additional_settings, data_tasks = train_hyperoptim(settings_optimization)

        truth_table["data_tasks"][i] = data_tasks

        # RESULTS
        print("Optimization results")
        scale_back_data = False  # True if you want to work with unscaled data
        if scale_back_data:
            decimals_quantile_score = 1
        else:
            decimals_quantile_score = 4

        # compute_metrics computes
        # self.pred_quantiles_test = None  # resulting predicted quantiles
        # self.data_reli_test = None  # data for plotting reliability plot
        # self.pred_conformal_df = None # dataframe with prediction for conformance analysis
        # self.task_params = None  # models parameters
        # self.test_dataloader = None  # test dataloader for prediction computation
        # self.pred_after_conformal_df = None  # predictions quantiles after conformance

        quantile_score_table, quantile_score_table_conformal = compute_metrics(
            data_tasks=data_tasks,
            settings_optimization=settings_optimization,
            additional_settings=additional_settings,
            settings_cp=settings_cp,
            scale_back_data=scale_back_data,
            decimals_quantile_score=decimals_quantile_score)

        truth_table["quantile_score"][i] = quantile_score_table.to_dict()
        truth_table["quantile_score_conformal"][i] = quantile_score_table_conformal.to_dict()

        with open(f"{main_path}/num_cali_samples_observed", "wb") as fp:  # save list with already observed samples
            pickle.dump(num_cali_samples_observed, fp)
    # plot results
    # overall results are contained in the truth_table, you can choose which row to plot
    row_to_display = 14
    data_tasks = truth_table["data_tasks"][row_to_display]

    for task_name in settings_optimization["task_names"]:
        _, _ = reliability_plot(
            quantiles=settings_optimization["quantiles"],
            y=data_tasks[task_name].pred_after_conformal_df['y'].to_numpy().reshape(-1, 1),
            pred_sorted=data_tasks[task_name].pred_after_conformal_df[settings_optimization["quantiles"]].values,
            task_name=task_name,
            plot_graph=True,
            conformal_title=True,
        )

        # reliability plot before conformal analysis
        samples_to_consider = len(data_tasks[task_name].data_te.index[
                                      data_tasks[task_name].data_te.index.year == data_tasks[task_name].year_te[
                                          0]])  # for a fair comparison of the quantile score

        _, _ = reliability_plot(
            quantiles=settings_optimization["quantiles"],
            y=data_tasks[task_name].pred_conformal_df['y'].to_numpy().reshape(-1, 1)[-samples_to_consider:],
            pred_sorted=data_tasks[task_name].pred_conformal_df[settings_optimization["quantiles"]].values[-samples_to_consider:],
            task_name=task_name,
            plot_graph=True,
            conformal_title=False
        )

        # sharpness diagram
        # plot_sharpness_plot(data_tasks[task_name].pred_conformal_df, task_name)
        # plot_sharpness_plot(data_tasks[task_name].pred_after_conformal_df, task_name) # conformal analysis

        # plot predictions quantiles after conformal
        plot_predictions_quantiles(
            task_name=task_name,
            quantiles=settings_optimization["quantiles"],
            y=data_tasks[task_name].pred_after_conformal_df['y'].to_numpy().reshape(-1, 1),
            pred_sorted=data_tasks[task_name].pred_after_conformal_df[settings_optimization["quantiles"]].values,
            conformal_analysis=True,
        )

        # plot predictions quantiles  before conformance
        plot_predictions_quantiles(
            task_name=task_name,
            quantiles=settings_optimization["quantiles"],
            y=data_tasks[task_name].pred_conformal_df['y'].to_numpy().reshape(-1, 1)[-samples_to_consider:],
            pred_sorted=data_tasks[task_name].pred_conformal_df[settings_optimization["quantiles"]].values[-samples_to_consider:],
            conformal_analysis=False
        )

    quantile_score_table = pd.DataFrame(data=truth_table["quantile_score"][row_to_display].values(), index=truth_table["quantile_score"][row_to_display].keys())[0]
    quantile_score_table_conformal = pd.DataFrame(data=truth_table["quantile_score_conformal"][row_to_display].values(), index=truth_table["quantile_score_conformal"][row_to_display].keys())[0]
    print(f"Quantile score before conformal: {quantile_score_table}, \n mean: {np.mean(quantile_score_table)}\n")
    print(f"Quantile score after conformal: {quantile_score_table_conformal}, \n mean: {np.mean(quantile_score_table_conformal)}\n")
    print(f"Quantile score variation (- if conformal is better): \n"
          f"{quantile_score_table_conformal-quantile_score_table} \n"
          f"mean: {np.mean(quantile_score_table_conformal-quantile_score_table)}")


    print("End execution")
