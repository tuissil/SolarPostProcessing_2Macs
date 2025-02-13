"""
Title: Post-processing of Solar Global Horizontal Irradiance through Conformalized Quantile Regression Neural Networks
@email: lorenzo.tuissi@stiima.cnr.it, alessandro.brusaferri@stiima.cnr.it
@author: Lorenzo Tuissi, Alessandro Brusaferri
"""

import numpy as np
import torch
from NWP_learning_utils import vali_model, DataloaderDataset, train_model_dataloader, PinballLoss, NWP_net, quantile_score, HuberNorm, create_folder, reliability_plot, plot_sharpness_plot, truth_table, train_hyperoptim, compute_metrics, compute_num_cali_samples, plot_predictions_quantiles_fill
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

    # target confidence levels
    target_alpha = [0.02, 0.1, 0.2, 0.4, 0.6, 0.8]

    # number of observations across each day
    pred_horiz = 9  # default 9

    # optimize pid: true, if you want to optimize the parameters of the pid and compute the performance on the last month of 2019;
    # false, if yu want to compute the performance over 2020
    optimize_pid = False


    # do not change following configurations
    if optimize_pid:
        test_months = [0]
        yr_va = [2019, 2020]
        yr_te = [2019, 2020]
    else:
        test_months = None
        days_cali_vali = None
        yr_va = [2019]
        yr_te = [2019, 2020]  # consider also cali days


    # create truth table with all the cases to analise
    methods = ['pid']  # you can choose between: ['cp', 'cqr', 'pid']
    step_wise_cp_options = [False]  # you can choose between: [True, False]
    num_cali_samples_true = [122]  # you can choose between: [50, 122, 150, 180], other values will start a new model retraining procedure  # if step_wise true
    num_cali_samples_false = [122]  # you can choose between: [15, 10, 5], other values will start a new model retraining procedure  # if not step_wise
    p_gains = [0.001] # 0.005 for stepwise, 0.001 for non-stepwise #[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] # only for pid method
    T_burnin_arr = [7, 180]  # ex. [7, 30, 180] maximum is internally imposed equal to num_cali

    # generate and initialize truth_table with all the cases
    truth_table = truth_table(methods, step_wise_cp_options, num_cali_samples_true, num_cali_samples_false, p_gains, optimize_pid, test_months, T_burnin_arr)
    truth_table["data_tasks"] = np.NaN
    truth_table["quantile_score"] = np.NaN
    truth_table["quantile_score_conformal"] = np.NaN

    # read which calibration dataset have been already trained
    num_cali_samples_observed = np.loadtxt(f"{main_path}/num_cali_samples_observed.txt", dtype=int)

    for i in range(truth_table.__len__()):
        cp_method = truth_table["method"][i]
        step_wise_cp = truth_table["step_wise_cp"][i]

        # do not change following lines
        if optimize_pid and test_months==[0]:
            num_cali_samples = 180
        else:
            num_cali_samples = compute_num_cali_samples(
                    step_wise_cp,
                    truth_table["num_cali_samples"][i],
                    target_alpha,
                    pred_horiz
            )

        # train a new model, if the required dataset splitting is not available
        if num_cali_samples not in num_cali_samples_observed:
            train_model_start = True
            num_cali_samples_observed = np.append(num_cali_samples_observed, num_cali_samples)
        else:
            train_model_start = False

        settings_cp = {
            'target_name': 'y',
            'num_cali_samples': num_cali_samples,  # if optimize pid, this value will be rewritten by settings_opt; number of last days of the validation set to use for the conformance analysis
            'target_alpha': target_alpha,
            'pred_horiz': pred_horiz,  # number of hours to predict
            'num_ense': 1,
            'stepwise': step_wise_cp,
            'cp_method': cp_method,
            'pid_lr': truth_table["p_gain"][i],
            'pid_KI': 0, # no integral action is considere
            'T_burnin': truth_table['T_burnin'][i],
        }
        print(f"Truth table row {i}, Method: {cp_method}, stepwise: {step_wise_cp}, num_cali_samples: {num_cali_samples}")

        #  Parameters setting for training and hyperoptimization

        quantiles = np.sort(np.concatenate([np.array(target_alpha) / 2, [0.5], 1 - np.array(target_alpha) / 2]))
        settings_optimization = {
            "task_names": ["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"],  #default: ["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"]
            "yr_tr": [2017, 2018],
            "yr_va": yr_va,
            "yr_te": yr_te,
            "scale_data": True,
            "shuffle_train": False,
            ## Optim params
            "quantiles": quantiles,
            "exp_hidden_size": 3,  # hidden_size = 1** exp_hidden_size, default 3
            "num_hidden_layers": 1,  # default 1
            "initial_seed": 0,  # default 0, same for both training and optimization of hyperparameters
            "main_path": main_path,
            "pred_horiz": pred_horiz,

            "train_model_start": train_model_start,  # retrain the model
            "exp_learning_rate": -4,  # default -4
            "num_epochs": 50,  # default 500
            "weight_decay_optimizer": 1e-3,  # default 1e-3, L2 penalty
            "decay_steps": 100,  # default = 100, for learning rate scheduler
            "error_precision": 4,  # default: 4, number of decimals in the error
            "batch_size": 16,  # default 16
            "shuffle_dataloader": True,
            "n_ensembles": 5,  # for ensemble NN
            "patience": 100,  # default 100
            "Newton_method": False,  # default False, set to True if you want to adopt a Newton method, not valid for hyperparam optim
            "criterion": PinballLoss(quantiles),  # HuberNorm(quantiles) #PinballLoss(quantiles)  # reduction='sum'

            # do not change following param
            "num_cali_samples": settings_cp['num_cali_samples'],  # number of days subtracted from yr_va in the validation set

            # hyperoptim set
            "hyper_param_opt": False,
            "num_epochs_hyperoptim": 150,  # default = 150
            "patience_hyperoptim": 50,  # default = 50
            "n_trials": 10,  # default = 10
            "pruner_epochs": 50,  # default = 50
            "pruner_max": 0.1,  # default = 0.1, if the computed error is above pruner_max, count patience
            "pruner_sensitivity": 1.2,  # default = 1.2, multiply the min_epoch_error for this factor to increase the patience counter

            # pid optimization
            "optimize_pid": optimize_pid,
            "days_cali": truth_table["num_cali_samples"][i], # if optimize pid, number of days used for calibration, valid just with optimize_pid and test_month = 0
            "test_set_number": truth_table["test_set_number"][i] # must be between 0 and 12, from December 2019 to December 2020
        }

        additional_settings, data_tasks = train_hyperoptim(settings_optimization)

        truth_table["data_tasks"][i] = data_tasks

        # RESULTS
        print("Optimization results")
        scale_back_data = True  # True if you want to work with unscaled data
        if scale_back_data:
            decimals_quantile_score = 4
        else:
            decimals_quantile_score = 4

        quantile_score_table, quantile_score_table_conformal = compute_metrics(
            data_tasks=data_tasks,
            settings_optimization=settings_optimization,
            additional_settings=additional_settings,
            settings_cp=settings_cp,
            scale_back_data=scale_back_data,
            decimals_quantile_score=decimals_quantile_score,
            optimize_pid=optimize_pid,
            test_month=truth_table["test_set_number"][i],
        )

        truth_table["quantile_score"][i] = quantile_score_table.to_dict()
        truth_table["quantile_score_conformal"][i] = quantile_score_table_conformal.to_dict()

        # save new dataset splitting
        np.savetxt(f"{main_path}/num_cali_samples_observed.txt", num_cali_samples_observed, fmt='%s')


    # plot results
    # overall results are contained in the truth_table, you can choose which row to plot

    # create a new column for each QS
    for stn in truth_table["quantile_score"][0].keys():
        truth_table[f"QS_{stn}"] = 0
        truth_table[f"QS_conf_{stn}"] = 0
        for i in range(truth_table.__len__()):
            truth_table[f"QS_{stn}"][i]=truth_table["quantile_score"][i][stn][0]
            truth_table[f"QS_conf_{stn}"][i] = truth_table["quantile_score_conformal"][i][stn][0]

    # compute mean along all the rows of the truth table
    truth_table.insert(len(truth_table.keys()), "Score deviation", np.NaN)
    truth_table.insert(len(truth_table.keys()), "Average_QS", np.NaN)
    truth_table.insert(len(truth_table.keys()), "Average_QS_conformal", np.NaN)

    for i in range(len(truth_table)):
        quantile_score_table = pd.DataFrame(data=truth_table["quantile_score"][i].values(),index=truth_table["quantile_score"][i].keys())[0]
        quantile_score_table_conformal = pd.DataFrame(data=truth_table["quantile_score_conformal"][i].values(), index=truth_table["quantile_score_conformal"][i].keys())[0]
        truth_table.at[i, "Score deviation"] = np.mean(quantile_score_table_conformal-quantile_score_table)
        truth_table.at[i, "Average_QS"] = np.mean(truth_table[['QS_bon','QS_dra', 'QS_fpk', 'QS_gwn', 'QS_psu', 'QS_sxf', 'QS_tbl']], axis=1)[i]
        truth_table.at[i, "Average_QS_conformal"] = np.mean(truth_table[['QS_conf_bon', 'QS_conf_dra', 'QS_conf_fpk', 'QS_conf_gwn', 'QS_conf_psu', 'QS_conf_sxf', 'QS_conf_tbl']], axis=1)[i]


    truth_table_display = truth_table[
        ['method', 'step_wise_cp', 'num_cali_samples', 'p_gain', 'test_set_number', 'Score deviation', "Average_QS", "Average_QS_conformal", "T_burnin"]]


    # save the truth table inn correct format for publications
    truth_table_for_pub = truth_table[['method', 'step_wise_cp', 'num_cali_samples', 'p_gain','T_burnin', "Average_QS", "Average_QS_conformal", 'Score deviation', 'QS_bon',
       'QS_dra', 'QS_fpk', 'QS_gwn', 'QS_psu', 'QS_sxf', 'QS_tbl', 'QS_conf_bon', 'QS_conf_dra', 'QS_conf_fpk', 'QS_conf_gwn', 'QS_conf_psu', 'QS_conf_sxf', 'QS_conf_tbl']]
    #truth_table_for_pub.to_csv(f"{main_path}\\truth_table_pid_122.csv", sep=";")


    truth_table.insert(len(truth_table.keys()), "PICP", np.NaN)
    truth_table.insert(len(truth_table.keys()), "PIAW", np.NaN)

    # which rows of the truth_table do you want to display?
    rows_to_display = [0, 1]
    save_plot = False  # save the plots locally
    plot_graphs = False  # show the plots
    initial_date_for_quantiles = "2020-11-23"  # initial date for quantile visualization
    final_date_for_quantiles = "2020-11-29"  # final date for quantile visualization


    for row_to_display in rows_to_display:
        data_tasks = truth_table["data_tasks"][row_to_display]
        PICP_before_conformal_ov = pd.DataFrame(index=quantiles, columns=settings_optimization["task_names"])
        PICP_after_conformal_ov=pd.DataFrame(index=quantiles, columns=settings_optimization["task_names"])
        PIAW_bc_ov = pd.DataFrame(index=[98, 90, 80], columns=settings_optimization["task_names"])
        PIAW_ac_ov = pd.DataFrame(index=[98, 90, 80], columns=settings_optimization["task_names"])
        for task_name in settings_optimization["task_names"]:

            # PICP after conformal
            _, PICP_after_conformal = reliability_plot(
                quantiles=settings_optimization["quantiles"],
                y=data_tasks[task_name].pred_after_conformal_df['y'].to_numpy().reshape(-1, 1),
                pred_sorted=data_tasks[task_name].pred_after_conformal_df[settings_optimization["quantiles"]].values,
                task_name=task_name,
                plot_graph=plot_graphs,
                conformal_title=True,
            )
            PICP_after_conformal_ov[task_name]=np.round(PICP_after_conformal['P1'].values*100,2)
            if save_plot:
                plt.gca().set_title('')
                plt.gca().get_legend().remove() # remove legend
                plt.gca().set_xlabel('')  # Remove the x-axis label
                plt.gca().set_ylabel('')  # Remove the y-axis label
                plt.gca().set_xticklabels('')
                plt.gca().set_yticklabels('')
                plt.savefig(f"{main_path}\\rel_plot\\{task_name}_conf_{rows_to_display.index(row_to_display)}_rel.eps", bbox_inches='tight', pad_inches=-0.1)

            # PICP before conformal
            # reliability plot before conformal analysis
            samples_to_consider = len(data_tasks[task_name].data_te.index[
                                          data_tasks[task_name].data_te.index.year == data_tasks[task_name].year_te[
                                              1]])  # for a fair comparison of the quantile score (computation over 2020)

            _, PICP_before_conformal = reliability_plot(
                quantiles=settings_optimization["quantiles"],
                y=data_tasks[task_name].pred_conformal_df['y'].to_numpy().reshape(-1, 1)[-samples_to_consider:],
                pred_sorted=data_tasks[task_name].pred_conformal_df[settings_optimization["quantiles"]].values[
                            -samples_to_consider:],
                task_name=task_name,
                plot_graph=plot_graphs,
                conformal_title=False
            )

            PICP_before_conformal_ov[task_name] = np.round(PICP_before_conformal['P1'].values * 100, 2)

            if rows_to_display.index(row_to_display)==0:
                if save_plot:
                    plt.gca().set_title('')
                    plt.gca().get_legend().remove() # remove legend
                    plt.gca().set_xlabel('')  # Remove the x-axis label
                    plt.gca().set_ylabel('')  # Remove the y-axis label
                    plt.gca().set_xticklabels('')
                    plt.gca().set_yticklabels('')
                    plt.savefig(f"{main_path}\\rel_plot\\{task_name}_base_{rows_to_display.index(row_to_display)}_rel.eps", bbox_inches='tight', pad_inches=-0.1)

            # predicted quantiles
            if rows_to_display.index(row_to_display) == 0:
                plot_predictions_quantiles_fill(
                    task_name=task_name,
                    quantiles=settings_optimization["quantiles"],
                    y=data_tasks[task_name].pred_conformal_df['y'][
                      initial_date_for_quantiles:final_date_for_quantiles].to_numpy().reshape(-1, 1),
                    pred_sorted=data_tasks[task_name].pred_conformal_df[settings_optimization["quantiles"]].loc[
                                initial_date_for_quantiles:final_date_for_quantiles].values,
                    conformal_analysis=False,
                    plot_graphs=plot_graphs
                )
                if save_plot:
                    plt.gca().set_title('')
                    plt.gca().get_legend().remove()  # remove legend
                    plt.gca().set_xlabel('')  # Remove the x-axis label
                    plt.gca().set_ylabel('')  # Remove the y-axis label
                    #plt.gca().set_xticklabels('')
                    #plt.gca().set_yticklabels('')
                    plt.savefig(
                        f"{main_path}\\pred_quan\\{task_name}_QRNN_quan.eps", bbox_inches='tight', pad_inches=0, transparent=True)

            # sharpness diagram for test = 2020, before conformal
            PIAW_98_bc, PIAW_90_bc, PIAW_80_bc = plot_sharpness_plot(data_tasks[task_name].pred_conformal_df[-samples_to_consider:], task_name, plot_graph=plot_graphs)
            PIAW_bc_ov[task_name]=np.round([PIAW_98_bc, PIAW_90_bc, PIAW_80_bc],2)
            if rows_to_display.index(row_to_display) == 0:
                if save_plot:
                    plt.gca().set_title('')
                    plt.gca().set_xlabel('')  # Remove the x-axis label
                    plt.gca().set_ylabel('')  # Remove the y-axis label
                    plt.gca().set_xticklabels('')
                    plt.gca().set_yticklabels('')
                    plt.savefig(f"{main_path}\\sharp_diag\\{task_name}_base_{rows_to_display.index(row_to_display)}_sharp.eps", bbox_inches='tight', pad_inches=-0.05)

            # sharpness diagram after conformal
            PIAW_98_ac, PIAW_90_ac, PIAW_80_ac = plot_sharpness_plot(data_tasks[task_name].pred_after_conformal_df, task_name, plot_graph=plot_graphs) # conformal analysis
            PIAW_ac_ov[task_name] = np.round([PIAW_98_ac, PIAW_90_ac, PIAW_80_ac], 2)

            if save_plot:
                plt.gca().set_title('')
                plt.gca().set_xlabel('')  # Remove the x-axis label
                plt.gca().set_ylabel('')  # Remove the y-axis label
                plt.gca().set_xticklabels('')
                plt.gca().set_yticklabels('')
                plt.savefig(f"{main_path}\\sharp_diag\\{task_name}_conf_{rows_to_display.index(row_to_display)}_sharp.eps", bbox_inches='tight', pad_inches=-0.05, transparent=True) # pad_inches=0, to not save border

            
            # plot predicted quantiles
            plot_predictions_quantiles_fill(
                task_name=task_name,
                quantiles=settings_optimization["quantiles"],
                y=data_tasks[task_name].pred_after_conformal_df['y'][initial_date_for_quantiles:final_date_for_quantiles].to_numpy().reshape(-1, 1),
                pred_sorted=data_tasks[task_name].pred_after_conformal_df[settings_optimization["quantiles"]].loc[initial_date_for_quantiles:final_date_for_quantiles].values,
                conformal_analysis=True,
                plot_graphs=plot_graphs
            )

            if save_plot:
                plt.gca().set_title('')
                plt.gca().get_legend().remove() # remove legend
                plt.gca().set_xlabel('')  # Remove the x-axis label
                plt.gca().set_ylabel('')  # Remove the y-axis label
                plt.savefig(
                    f"{main_path}\\pred_quan\\{task_name}_conf_{rows_to_display.index(row_to_display)}_quan.eps",
                    bbox_inches='tight', pad_inches=0, transparent=True)

            # plot quantiles after QRNN
            plot_predictions_quantiles_fill(
                task_name=task_name,
                quantiles=settings_optimization["quantiles"],
                y=data_tasks[task_name].pred_after_conformal_df['y'][initial_date_for_quantiles:initial_date_for_quantiles].to_numpy().reshape(-1, 1),
                pred_sorted=data_tasks[task_name].pred_conformal_df[settings_optimization["quantiles"]].loc[initial_date_for_quantiles:initial_date_for_quantiles].values,
                conformal_analysis=False,
                plot_graphs=plot_graphs
            )

            '''
            # plot the component forecasts
            ensemble_plot = data_tasks[task_name].data_original_df[initial_date_for_quantiles:initial_date_for_quantiles]
            E_columns = [f"E_{i}" for i in np.sort(np.random.randint(0, 50, 3))]
            for col in E_columns:
                plt.figure(figsize=(10, 2))
                plt.plot(ensemble_plot.index, ensemble_plot[col], label=col, alpha=0.7)
                plt.ylabel(r"GHI [W/m$^2$]")
                plt.title(col)
                plt.grid()
            plt.show()
            '''

        # save PICP in truth_table
        PICP_before_conformal_ov['avg'] = np.round((np.mean(PICP_before_conformal_ov, axis=1)), 2).values
        PICP_after_conformal_ov['avg'] = np.round((np.mean(PICP_after_conformal_ov, axis=1)), 2).values

        truth_table["PICP"][row_to_display] = {
            "PICP_before_conformal": PICP_before_conformal_ov,
            "PICP_after_conformal": PICP_after_conformal_ov,
        }

        # save PIAW in truth_table
        PIAW_bc_ov['avg'] = np.round((np.mean(PIAW_bc_ov, axis=1)), 2).values
        PIAW_ac_ov['avg'] = np.round((np.mean(PIAW_ac_ov, axis=1)), 2).values

        truth_table["PIAW"][row_to_display] = {
                f"PIAW_before_conformal": PIAW_bc_ov,
                f"PIAW_after_conformal": PIAW_ac_ov,
        }

    print("End execution")
