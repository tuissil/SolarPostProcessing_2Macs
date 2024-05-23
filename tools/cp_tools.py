"""
Conformal prediction utils
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

from typing import Dict, List
import numpy as np
import sys
from tools.cp_pi import cts_pid

def build_alpha_quantiles_map(target_alpha: List, target_quantiles: List):
    """
    Build the map between PIs coverage levels and related quantiles
    """
    alpha_q = {'med': target_quantiles.index(0.5)}
    for alpha in target_alpha:
        alpha_q[alpha] = {
            'l': target_quantiles.index(alpha / 2),
            'u': target_quantiles.index(1 - alpha / 2),
        }
    return alpha_q

def build_target_quantiles(target_alpha):
    """
    Build target quantiles from the alpha list
    """
    target_quantiles = [0.5]
    for alpha in target_alpha:
        target_quantiles.append(alpha / 2)
        target_quantiles.append(1 - alpha / 2)
    target_quantiles.sort()
    return target_quantiles


def fix_quantile_crossing(preds: np.array):
    """
    Fix crossing in the predicted quantiles by means of post-hoc sorting
    """
    return np.sort(preds, axis=-1)

def exec_cqr(preds_cali: np.array, y_cali: np.array, preds_test: np.array, settings: Dict):
    """
       Compute conformalized prediction intervals:
    """

    def __asym_mode__(conf_scores: np.array, alpha:float):
        """
        Function to compute asymmetric cqr
        """
        q=(1 - alpha / 2)
        Q_l = np.quantile(a=conf_scores[:, :, 0], q=q, axis=0, method='higher')
        Q_h = np.quantile(a=conf_scores[:, :, 1], q=q, axis=0, method='higher')
        return np.stack([Q_l, Q_h], axis=-1)

    def __conform_prediction_intervals__(pred_pi: np.array, q_cp: np.array):
        """
        Function to compute conformalized PIs
        """
        return np.concatenate([np.expand_dims(pred_pi[:, :, 0] - q_cp[:, 0], axis=2),
                               np.expand_dims(pred_pi[:, :, 1] + q_cp[:, 1], axis=2)], axis=2)

    # map the method to be employed to conformalize
    compute_score_quantiles = __asym_mode__

    if not settings['stepwise']:
        preds_cali=preds_cali.reshape(-1, 1, preds_cali.shape[2])
        y_cali = y_cali.reshape(-1,1)

    # Conformalize predictions for each alpha
    for alpha in settings['target_alpha']:
        # get index of the lower/upper quantiles for the current alpha from the map
        lq_idx = settings['q_alpha_map'][alpha]['l']
        uq_idx = settings['q_alpha_map'][alpha]['u']

        # Compute conformity scores according to [1]-(6)
        conf_scores = np.stack([preds_cali[:, :, lq_idx] - y_cali, y_cali - preds_cali[:, :, uq_idx]], axis=-1)
        conformalized_alpha_PIs= __conform_prediction_intervals__(pred_pi=preds_test[:,:,[lq_idx, uq_idx]],
                                                                  q_cp=compute_score_quantiles(conf_scores=conf_scores,
                                                                                               alpha=alpha))
        # replace test_pred PIs columns with conformalized PIs for each alpha
        preds_test[:, :, lq_idx] = conformalized_alpha_PIs[:,:,0]
        preds_test[:, :, uq_idx] = conformalized_alpha_PIs[:,:,1]

    # Fix quantile crossing
    # return prediction flattened in temporal dimension (sample over pred horizon)
    preds_test[preds_test < 0] = 0
    return fix_quantile_crossing(preds_test.reshape(-1, preds_test.shape[-1]))

def compute_cqr(results_e, settings: Dict):
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    # aggregate ensemble quantiles, fix crossimg, and compute cqr
    agg_q = []
    for t_q in settings['target_quantiles']:
        q_c = []
        for e_c in range(settings['num_ense']):
            q_c.append(results_e[e_c].loc[:, t_q].to_numpy().reshape(-1, 1))
        agg_q.append(np.mean(np.concatenate(q_c, axis=1), axis=1).reshape(-1, 1))
    q_ens = fix_quantile_crossing(np.concatenate(agg_q, axis=1))

    target_d = results_e[0].filter([settings['target_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'], )
    preds_d = q_ens.reshape(-1, settings['pred_horiz'], q_ens.shape[-1])
    num_test_samples = preds_d.shape[0] - settings['num_cali_samples']
    settings['cp_options']= {'cqr_mode': 'asym'},
    settings['q_alpha_map']= build_alpha_quantiles_map(target_quantiles=settings['target_quantiles'],
                                                       target_alpha=settings['target_alpha'])
    test_PIs = []
    for t_s in range(num_test_samples):
        preds_cali = preds_d[t_s:settings['num_cali_samples'] + t_s]
        preds_test = preds_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s + 1]
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]

        test_PIs.append(exec_cqr(preds_cali=preds_cali,
                                 y_cali=y_cali,
                                 preds_test=preds_test,
                                 settings=settings))

    test_PIs = np.concatenate(test_PIs, axis=0)
    aggr_df = results_e[0].filter([settings['target_name']], axis=1)
    aggr_df = aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]
    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]] = test_PIs[:, j]
    return aggr_df


def compute_pid(results_e, settings, lr=0.01, KI=10, T_burnin=7, Tin=1e9, delta=5e-2):
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    # aggregate ensemble quantiles, fix crossimg, and compute cqr
    agg_q = []
    for t_q in settings['target_quantiles']:
        q_c = []
        for e_c in range(settings['num_ense']):
            q_c.append(results_e[e_c].loc[:, t_q].to_numpy().reshape(-1, 1))
        agg_q.append(np.mean(np.concatenate(q_c, axis=1), axis=1).reshape(-1, 1))
    q_ens = fix_quantile_crossing(np.concatenate(agg_q, axis=1))

    q_alpha_map = build_alpha_quantiles_map(target_quantiles=settings['target_quantiles'],
                                            target_alpha=settings['target_alpha'])

    df = results_e[0].filter([settings['target_name']], axis=1)

    # initialize the output quantiles with the median
    pred_q = {q_alpha_map['med']: q_ens[:, q_alpha_map['med']]}
    for alpha in settings['target_alpha']:
        # get index of the lower/upper quantiles for the current alpha from the map
        lq_idx = q_alpha_map[alpha]['l']
        uq_idx = q_alpha_map[alpha]['u']
        sets_h = []

        if settings['stepwise']:
            preds_d = q_ens.reshape(-1, settings['pred_horiz'], q_ens.shape[-1])
            eh=df.index.min().hour
            lh = df.index.max().hour
            h_j=0
            for h in range(eh, lh+1):
                t_s = str(h)
                t_0 = t_s + ':00:00'
                t_1 = t_s + ':00:30'
                df_h = df.between_time(t_0, t_1).copy()
                df_h.rename(columns={settings['target_name']: 'y'}, inplace=True)
                df_h['forecasts'] = [np.array([preds_d[j, h_j, lq_idx], preds_d[j, h_j, uq_idx]])
                                     for j in range(len(df_h))]

                Csat = (2 / np.pi) * (np.ceil(np.log(Tin) * delta) - (1 / np.log(Tin)))

                results = cts_pid(data=df_h, alpha=alpha, lr=lr, Csat=Csat, KI=KI, T_burnin=settings['T_burnin'])
                sets_h.append(np.stack(results['sets'], axis=1).T)
                h_j+=1
        else:
            preds_d = q_ens
            df.rename(columns={settings['target_name']: 'y'}, inplace=True)
            df['forecasts'] = [np.array([preds_d[j, lq_idx], preds_d[j, uq_idx]])
                                 for j in range(len(df))]

            Csat = (2 / np.pi) * (np.ceil(np.log(Tin) * delta) - (1 / np.log(Tin)))

            results = cts_pid(data=df, alpha=alpha, lr=lr, Csat=Csat, KI=KI, T_burnin=T_burnin)
            sets_h.append(np.stack(results['sets'], axis=1).T)


        sets_p = np.stack(sets_h, axis=1).reshape(-1, 2)
        pred_q[q_alpha_map[alpha]['l']] = sets_p[:, 0]
        pred_q[q_alpha_map[alpha]['u']] = sets_p[:, 1]

    test_q = dict(sorted(pred_q.items()))
    test_q = np.array(list(test_q.values())).T
    test_q = fix_quantile_crossing(test_q)
    test_q[test_q < 0] = 0
    results_df = results_e[0].filter([settings['target_name']], axis=1)
    for j in range(len(settings['target_quantiles'])):
        results_df[settings['target_quantiles'][j]] = test_q[:, j]
    return results_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]


def exec_cp(preds_cali: np.array, y_cali: np.array, preds_test: np.array, settings: Dict):
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0]>1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    n=conf_score.shape[0]
    # Stack the quantiles to the point pred for each alpha (next sorted by fixing crossing)
    preds_test_q=[preds_test]
    for alpha in settings['target_alpha']:
        q = min(1, np.ceil((n + 1) * (1 - alpha)) / n)  ######### TO DO
        Q_1_alpha= np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method='higher'), axis=(0,-1))
        # Append lower/upper PIs for the current alpha
        preds_test_q.append(preds_test - Q_1_alpha)
        preds_test_q.append(preds_test + Q_1_alpha)
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    preds_test_q[preds_test_q < 0] = 0
    # Fix quantile crossing
    # return prediction flattened in temporal dimension (sample over pred horizon)
    return fix_quantile_crossing(preds_test_q.reshape(-1, preds_test_q.shape[-1]))


def compute_cp(results_e, settings):
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    # compute cp from point preds, fix crossing
    ens_p = []
    for e_c in range(settings['num_ense']):
        ens_p.append(results_e[e_c].loc[:, 0.5].to_numpy().reshape(-1, 1))
    ens_p = np.mean(np.concatenate(ens_p, axis=1), axis=1)
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    target_d = results_e[0].filter([settings['target_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])
    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    test_PIs = []
    for t_s in range(num_test_samples):
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s + 1]
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
        if not settings['stepwise']:
            preds_cali = preds_cali.reshape(-1, 1)
            preds_cali = np.expand_dims(preds_cali, axis=-1)
            y_cali = y_cali.reshape(-1,1)

        test_PIs.append(exec_cp(preds_cali=preds_cali,
                                y_cali=y_cali,
                                preds_test=preds_test,
                                settings={'target_alpha': settings['target_alpha']}))

    test_PIs = np.concatenate(test_PIs, axis=0)
    aggr_df = results_e[0].filter([settings['target_name']], axis=1)
    aggr_df = aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]
    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]] = test_PIs[:, j]
    return aggr_df



