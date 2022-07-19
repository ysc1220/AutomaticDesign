# Combine optimize_ann.py in hjkgrp github
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import hyperopt as ho
import pickle

from sklearn.model_selection import train_test_split

from functools import partial
from datetime import datetime

from molSimplifyAD.retrain.nets import build_ANN
from molSimplifyAD.retrain.model_optimization import train_model_hyperopt
from molSimplifyAD.logger import Logger

def drop_features(ml):
    fnames_drop =   []
    for fname in ml.fnames:
        std =   np.std(ml.df[str(fname)].values)
        if std < 1e-6:
            fnames_drop.append(fname)

    if len(fnames_drop) == 0:
        ml.info("None of the features is removed.")
    else:
        ml.info("# Removed feature:")
        for fname in fnames_drop:
            ml.fnames.remove(fname)
            ml.info("\t"+fname)
    ml.info("# Number of features: %d"%len(ml.fnames))

def normalize(ml):
    X_train =   ml.df_train[ml.fnames].values
    X_val   =   ml.df_val[ml.fnames].values
    y_train =   ml.df_train[ml.lnames].values
    y_val   =   ml.df_val[ml.lnames].values

    ml.X_scaler =   sk.preprocessing.StandardScaler().fit(X_train)
    ml.X_train  =   ml.X_scaler.transform(X_train)
    ml.X_val    =   ml.X_scaler.transform(X_val)
    ml.y_scaler =   sk.preprocessing.StandardScaler().fit(y_train)
    ml.y_train  =   ml.y_scaler.transform(y_train)
    ml.y_val    =   ml.y_scaler.transform(y_val)

    ml.info("Normalization finished.\n")

def train_ann(ml, params):
    X_train =   np.array(ml.X_train)
    X_val   =   np.array(ml.X_val)
    y_train =   np.array(ml.y_train)
    y_val   =   np.array(ml.y_val)

    model   =   build_ANN(params,
                          input_len = X_train.shape[-1],
                          lname = ml.lnames,
                          regression = ml.regression)

    history =   model.fit(X_train, y_train,
                          epochs = params["epochs"],
                          verbose = 2,
                          batch_size = params["batch_size"],
                          validation_data = (X_val, y_val))

    return model

def evaluate_ann(ml, model):
    X_train =   np.array(ml.X_train)
    X_val   =   np.array(ml.X_val)
    y_train =   np.array(ml.y_train)
    y_val   =   np.array(ml.y_val)

    res_train   =   model.evaluate(X_train, y_train)
    res_val     =   model.evaluate(X_val, y_val)
    res_train_dict  =   {}
    res_val_dict    =   {}
    for i, key in enumerate(model.metrics_names):
        res_train_dict[key] =   res_train[i]
        res_val_dict[key]   =   res_val[i]

    y_train_true    =   ml.y_scaler.inverse_transform(y_train)
    y_val_true      =   ml.y_scaler.inverse_transform(y_val)
    y_train_pred    =   ml.y_scaler.inverse_transform(
                                model.predict(X_train))
    y_val_pred      =   ml.y_scaler.inverse_transform(
                                model.predict(X_val))

    res_train_dict["mae_org"]   =   sk.metrics.mean_absolute_error(
                                    y_train_true, y_train_pred)
    res_val_dict["mae_org"]     =   sk.metrics.mean_absolute_error(
                                    y_val_true, y_val_pred)
    res_train_dict["r2_score"]  =   sk.metrics.r2_score(
                                    y_train_true, y_train_pred)
    res_val_dict["r2_score"]    =   sk.metrics.r2_score(
                                    y_val_true, y_val_pred)

    ml.info("# Model performance")
    for key in res_train_dict:
        ml.info("\t%s: train - %f\tval - %f"%(key,
                res_train_dict[key], res_val_dict[key]))

    return res_train_dict, res_val_dict

def optimize_ann(ml):
    np.random.seed(1234)
    ml.nit  =   0

    archs   =   [(128, 128),
                 (256, 256),
                 (512, 512),
                 (128, 128, 128),
                 (256, 256, 256),
                 (512, 512, 512)]
    bzs =   [16, 32, 64, 128, 256]
    ress    =   [True, False]
    bypasses    =   [True, False]

    space   =   {"lr":  ho.hp.uniform("lr", 1e-5, 1e-3),
                 "drop_rate":   ho.hp.uniform("drop_rate", 0, 0.5),
                 "reg": ho.hp.loguniform("reg", np.log(1e-5), np.log(1e0)),
                 "batch_size":  ho.hp.choice("batch_size", bzs),
                 "hidden_size": ho.hp.choice("hidden_size", archs),
                 "beta_1":  ho.hp.uniform("beta_1", 0.80, 0.99),
                 "decay":   ho.hp.loguniform("decay", np.log(1e-5), np.log(1e0)),
                 "res": ho.hp.choice("res", ress),
                 "bypass":  ho.hp.choice("bypass", bypasses),
                 "amsgrad": True,
                 "patience":    100}

    objective_func  =   partial(train_model_hyperopt,
                                ml = ml,
                                save_model = False)
                                #X   =   np.array(ml.X_train),
                                #y   =   np.array(ml.y_train),
                                #lname   =   ml.lnames,
                                #regression  =   True,
                                #epochs  =   1000,
                                #X_val   =   np.array(ml.X_val),
                                #y_val   =   np.array(ml.y_val),
                                #input_model =   False)

    cput0   =   ml.set_timer()
    trials  =   ho.Trials()
    best_params =   ho.fmin(objective_func,
                            space,
                            algo    =   ho.tpe.suggest,
                            trials  =   trials,
                            max_evals   =   ml.max_evals,
                            rstate  =   np.random.default_rng(0))

    for key, ls in [("hidden_size", archs),
                    ("batch_size", bzs),
                    ("res", ress),
                    ("bypass", bypasses)]:
        best_params[key]    =   ls[best_params[key]]
    best_params["amsgrad"]  =   True
    best_params["patience"] =   100

    res =   train_model_hyperopt(best_params,
                                 ml = ml,
                                 save_model = True)
                                 #np.array(ml.X_train),
                                 #np.array(ml.y_train),
                                 #ml.lnames,
                                 #regression =   True,
                                 #epochs =   1000,
                                 #X_val  =   np.array(ml.X_val),
                                 #y_val  =   np.array(ml.y_val),
                                 #input_model    =   False)

    best_params["epochs"]   =   res["epochs"]
    ml.best_params  =   best_params

    with open(ml.filname_trials, "wb") as fil:
        pickle.dump(trials, fil)

    ml.timer("Hyperopt", *cput0)
    ml.info("# Best params:")
    for key, val in best_params.items():
        ml.info("\t%s:\t%s"%(key, str(val)))

    return trials

class ML(Logger):
    df  =   None
    fnames  =   None
    lnames  =   None

    regression  =   True
    epochs  =   1000
    input_model =   False
    max_evals   =   100

    filname_trials  =   "trials.pkl"
    filname_model   =   "model.h5"

    verbose =   8
    stdout  =   sys.stdout

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        if type(self.df) != type(None):
            self.df_train, self.df_val  =   train_test_split(
                                        self.df,
                                        test_size   =   0.2,
                                        random_state    =   42)
            self.dump_flags()

    def dump_flags(self):
        self.info("******** %s ********", self.__class__)
        self.info("******** %s ********",
            datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
        self.info("# Number of entries: train - %d\tval - %d"%(
                len(self.df_train), len(self.df_val)))

    def run(self):
        self.drop_features()
        self.normalize()
        self.optimize_ann()
        self.evaluate_ann()

    drop_features   =   drop_features
    normalize       =   normalize
    train_ann       =   train_ann
    optimize_ann    =   optimize_ann
    evaluate_ann    =   evaluate_ann

