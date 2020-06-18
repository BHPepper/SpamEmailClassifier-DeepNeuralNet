# -*- coding: utf-8 -*-
"""
Benjamin H Pepper
B.H.Pepper@gmail.com
https://www.linkedin.com/in/benjamin-pepper-62936714b/
"""

import numpy as np
from sklearn.model_selection import KFold
import statistics
import pandas as pd

def nested_cv(model, X, y, score_f, outer_perf_f,
              perf_type = 'low', grid = None, 
              inner_perf_f = None, cv_k1 = 10, cv_k2 = 10, 
              seed = 1, best_perf_method = 'mean'):
    np.random.seed(seed)
    kf_outer = KFold(n_splits=cv_k1, shuffle=True, random_state=seed)
    outer_perf = []
    scores = []
    responses = []
    best_params = []
    for train_out, test_out in kf_outer.split(X):
        kf_inner = KFold(n_splits=cv_k2, shuffle=True, random_state=seed)
        if grid is not None:
            inner_perf = []
            X_out = X.iloc[train_out,:]
            y_out = y[train_out]
            for train_in, test_in in kf_inner.split(X_out):
                perf_rows = []
                for row in range(np.shape(grid)[0]):
                    fit = model(X_out.iloc[train_in,:], y_out[train_in], seed, grid.iloc[row,:])
                    perf_rows.append(inner_perf_f(fit, X_out.iloc[test_in,:], y_out[test_in]))
                inner_perf.append(perf_rows)
            inner_perf = pd.DataFrame(inner_perf)
            if best_perf_method == 'mean':
                if perf_type == 'low':
                    best = inner_perf.mean().idxmin()
                else:
                    best = inner_perf.mean().idxmax()
            else:
                if perf_type == 'low':
                    statistics.mode(inner_perf.idxmin())
                else:
                    statistics.mode(inner_perf.idxmax())
            fit = model(X.iloc[train_out,:], y[train_out], seed, grid.iloc[best,:])
            outer_perf.append(outer_perf_f(fit, X.iloc[test_out,:], y[test_out]))
            scores.append(score_f(fit, X.iloc[test_out,:], y[test_out]))
            responses.append(y[test_out])
            best_params.append(best)
        else:   
            fit = model(X.iloc[train_out,:], y[train_out], seed)
            outer_perf.append(outer_perf_f(fit, X.iloc[test_out,:], y[test_out]))
            scores.append(score_f(fit, X.iloc[test_out,:], y[test_out]))
            responses.append(y[test_out])
            best_params.append(None)
    res = {'Perf': outer_perf, 'BestParams': best_params, 'Scores': np.concatenate(np.array(scores)), 'Y': np.concatenate(np.array(responses))}
    return(res)
