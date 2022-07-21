# 071922 copied rfa_krr.py in hjkgrp github
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import hyperopt as ho
import pickle
import json

import operator
from keras import backend as K
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from functools import partial

from molSimplifyAD.ML import ML

def train_rf(ml):
    cput0   =   ml.set_timer()

    params  =   ml.params_rf.copy()
    reg =   RandomForestRegressor(**params)

    reg.fit(ml.X_train, ml.y_train)
    oob_score   =   reg.oob_score_
    reg.score(ml.X_train, ml.y_train)
    pred    =   reg.oob_prediction_
    mae =   sk.metrics.mean_absolute_error(ml.y_train, pred)

    ml.info("# OOB_SCORE: %f", oob_score)
    ml.info("# OOB_MAE: %f", mae)
    ml.timer("Random forest", *cput0)

    return reg, mae

def initial_feature_ranking(ml):
    reg, mae    =   ml.train_rf()
    fis     =   reg.feature_importances_
    max_fi  =   max(fis)

    fscore_dict =   {}
    fnames_selected =   []

    for i, fname in enumerate(ml.fnames):
        fscore_dict[fname]  =   round(fis[i], 4)
        if fis[i] > ml.cutoff*max_fi:
            fnames_selected.append(fname)

    sorted_x    =   sorted(fscore_dict.items(),
                           key = operator.itemgetter(1),
                           reverse = True)

    fnames_top25    =   [x[0] for x in sorted_x[:25]]
    fnames_selected  =   set(fnames_selected).intersection(set(fnames_top25))

    ml.fnames_sorted   =   [elem[0] for elem in sorted_x]
    ml.fscore_dict  =   fscore_dict

    if hasattr(ml, "filname_fi_full"):
        with open(ml.filname_fi_full, "w") as fil:
            json.dump(fscore_dict, fil, index = 4)
        ml.info("Feature importance of full features written in %s", ml.filname_fi_full)

    ml.debug("# Initial features: %d", len(fnames_selected))
    i   =   0
    ml.fnames_selected  =   []
    for fname in ml.fnames_sorted:
        if fname in fnames_selected:
            ml.fnames_selected.append(fname)
            ml.debug("\t%d\t%s\t%f", i, fname, fscore_dict[fname])
            i   +=  1

    return ml.fnames_selected, ml.fnames_sorted

def optimize_features(ml):
    ml.fnames   =   ml.fnames_selected.copy()
    ml.info("# Number of initial features: %d", len(ml.fnames))

    fnames_sorted   =   []
    for fname in ml.fnames_sorted:
        if not fname in ml.fnames:
            fnames_sorted.append(fname)

    ml.normalize()
    best_params =   ml.optimize()
    model   =   ml.train(best_params)
    r2, mae =   ml.evaluate(model)
    K.clear_session()

    best_mae    =   mae
    best_r2 =   r2
    nit =   1
    ml.info("******** Feature selection ********")
    ml.info("Current best: MAE - %f r2 - %f", best_mae, best_r2)
    cput1   =   ml.set_timer()

    added_feature   =   ml.fnames.copy()
    while len(added_feature) > 0:
        added_feature   =   []
        ml.info("# Outer iteration %d", nit)
        ml.info("\tCurrent number of sorted features %d",
                len(fnames_sorted))
        cput0   =   ml.set_timer()

        for i, fname in enumerate(fnames_sorted):
            ml.debug("\t\tInner iteration %d / %d", i, len(fnames_sorted))
            ml.debug("\t\tEvaluating the feature "+fname)
            ml.fnames.append(fname)

            ml.normalize()
            best_params =   ml.optimize()
            model       =   ml.train(best_params)
            r2, mae     =   ml.evaluate(model)
            K.clear_session()

            ml.debug("\t\tMAE - %f r2 - %f", mae, r2)

            if mae < best_mae*0.99:
                best_mae    =   mae
                best_r2     =   r2
                added_feature.append(fname)
                ml.debug("\t\tFeature %s added", fname)
            else:
                ml.fnames.remove(fname)
                ml.debug("\t\tFeature %s discarded", fname)

            ml.debug("\t\tCurrent best: MAE - %f r2 - %f", best_mae, best_r2)

        for fname in added_feature:
            ml.fnames_selected.append(fname)
            fnames_sorted.remove(fname)

        ml.info("\tCurrent best: MAE - %f r2 - %f", best_mae, best_r2)
        ml.timer("Outer iteration %d"%nit, *cput0)
        nit +=  1

    ml.info("# Number of selected features: %d", len(ml.fnames))
    fscore_dict =   {}
    for i, fname in enumerate(ml.fnames):
        ml.info("\t%d\t%s\t%f", i, fname, ml.fscore_dict[fname])
        fscore_dict[fname]  =   ml.fscore_dict[fname]
    if hasattr(ml, "filname_fi_selected"):
        with open(ml.filname_fi_selected, "w") as fil:
            json.dump(fscore_dict, fil, indent = 4)
        ml.info("Feature importance of selected features written in %s", ml.filname_fi_selected)

    ml.timer("Feature selection", *cput1)
    ml.info("******** End of feature selection ********")

    return best_mae, best_r2

def train(ml, params, save_model = False):
    params["kernel"]    =   "rbf"
    model   =   KernelRidge(**params)
    model.fit(ml.X_train, ml.y_train)

    if save_model:
        with open(ml.filname_model, "wb") as fil:
            pickle.dump(model, fil)
        ml.info("Model saved to %s", ml.filname_model)

    return model

def train_hyperopt(params, ml):
    model   =   ml.train(params)

    y_pred  =   model.predict(ml.X_val)
    mae =   sk.metrics.mean_absolute_error(ml.y_val, y_pred)
    K.clear_session()

    return mae

def optimize(ml):
    objective_func  =   partial(train_hyperopt, ml = ml)
    trials  =   ho.Trials()
    best_params =   ho.fmin(objective_func,
                     ml.space,
                     algo   =   ho.tpe.suggest,
                     trials =   trials,
                     max_evals  =   ml.max_evals,
                     rstate =   np.random.default_rng(0),
                     verbose    =   0)

    return best_params

def evaluate(ml, model):
    y_train_pred    =   ml.y_scaler.inverse_transform(
                        model.predict(ml.X_train)).reshape(-1, )
    y_val_pred      =   ml.y_scaler.inverse_transform(
                        model.predict(ml.X_val)).reshape(-1, )
    y_train_true    =   ml.y_scaler.inverse_transform(
                        ml.y_train).reshape(-1, )
    y_val_true      =   ml.y_scaler.inverse_transform(
                        ml.y_val).reshape(-1, )

    if hasattr(ml, "filname_y"):
        with open(ml.filname_y, "wb") as fil:
            pickle.dump({"y_train_true": list(y_train_true),
                       "y_val_true": list(y_val_true),
                       "y_train_pred": list(y_train_pred),
                       "y_val_pred": list(y_val_pred)}, fil)
        ml.info("y values written in %s", ml.filname_y)

    return sk.metrics.r2_score(y_val_true, y_val_pred), \
            np.mean(np.abs(y_val_true-y_val_pred))

class RFA_KRR(ML):
    cutoff  =   0.05    # cutoff for feature selection

    params_rf   =   {"n_estimators":    100,
                     "criterion":   "mae",
                     "max_depth":   None,
                     "min_samples_leaf":    1,
                     "max_features": "auto",
                     "random_state":    np.random.RandomState(0),
                     "oob_score":   True}

    space   =   {"alpha":   ho.hp.loguniform("alpha",
                            np.log(1e-8), np.log(1e2)),
                 "gamma":   ho.hp.loguniform("gamma",
                            np.log(1e-8), np.log(1e2))}

    filname_model   =   "model.pkl"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        ML.__init__(self)

    def run(self):
        cput0   =   self.set_timer()

        self.drop_features()
        self.normalize()

        self.initial_feature_ranking()
        self.optimize_features()

        self.info("******** Final training ********")
        best_params =   self.optimize()
        model   =   self.train(best_params, save_model = True)
        r2, mae =   self.evaluate(model)

        self.info("# Best_params:")
        for key, val in best_params.items():
            self.info("\t%s:\t%s"%(key, str(val)))
        self.info("# Model performance")
        self.info("\tr2: %f", r2)
        self.info("\tmae: %f", mae)

        self.timer("RFA-KRR", *cput0)

    train_rf    =   train_rf
    initial_feature_ranking =   initial_feature_ranking
    optimize_features       =   optimize_features
    train       =   train
    optimize    =   optimize
    evaluate    =   evaluate

