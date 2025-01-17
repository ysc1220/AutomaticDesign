import sys
import numpy as np
from functools import partial
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, rand
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import backend as K
from molSimplifyAD.retrain.nets import build_ANN, auc_callback, cal_auc, compile_model
from keras import backend as K
#K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)))


def train_model_hyperopt(hyperspace, ml, save_model = False):
    '''
    Train a NN model and return information required in hyperopt

    Parameters
    ---
        hyperspace: dict, possible hyperparameter space
        X: np.array, features or inputs
        y: np.array, targets
        lname: str, name of the target property
        regression: boolean, whether it is a regression task
        epochs: int, maximum epochs during the model training
        X_val: boolean or np.array, validation features
        y_val: boolean or np.array, validation targets
        input_model: boolean for keras.model, can start with a input model instead of building from scratch.

    Returns
    ---
        a dictionary that hyperopt uses to proceed its Bayesian optimization.
    '''
    X   =   np.array(ml.X_train)
    y   =   np.array(ml.y_train)
    X_val   =   np.array(ml.X_val)
    y_val   =   np.array(ml.y_val)
    lname   =   ml.lnames
    regression  =   ml.regression
    epochs  =   ml.epochs
    input_model =   ml.input_model

    np.random.seed(1234)
    if tf.__version__ >= tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()  ## disable eager in tf2.0 for faster training

    ml.nit  +=  1
    ml.info("# Iteration %d"%ml.nit)
    for key, val in hyperspace.items():
        ml.info("\t%s %s"%(key, str(val)))
    cput0   =   ml.set_timer()

    if input_model == False:
        model = build_ANN(hyperspace, X.shape[-1], lname, regression=regression)
    else:
        model = compile_model(model=input_model, hyperspace=hyperspace,
                              lname=lname, regression=regression)
    if (isinstance(X_val, bool)) and (isinstance(y_val, bool)):
        X_train, X_val = np.split(X, [int(0.8 * X.shape[0])])
        y_train, y_val = np.split(y, [int(0.8 * X.shape[0])])
        print(X_train.shape, y_train.shape)
        y_train, y_val = list(np.transpose(y_train)), list(np.transpose(y_val))
    elif (isinstance(X_val, bool)) or (isinstance(y_val, bool)):
        raise ValueError("Both X_val and y_val need to be specified!")
    else:
        X_train, y_train = X, y
    val_data = (X_val, y_val)
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0.0,
                              patience=hyperspace['patience'],
                              verbose=0)
    cb = [earlystop]
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        verbose=0,
                        batch_size=hyperspace['batch_size'],
                        validation_data=val_data,
                        callbacks=cb,)
    if regression:
        obj = 0
        if len(lname) > 1:
            for ii, ln in enumerate(lname):
                key = "val_output-%d-%s_scaled_mae" % (ii, ln)
                obj += history.history[key][-1]
            obj /= len(lname)
        else:
            if sys.version_info[0] > 2:
                obj = history.history['val_mae'][-1]
            else:
                obj = history.history['val_mean_absolute_error'][-1]
    else:
        if len(lname) > 1:
            for ii, ln in enumerate(lname):
                key = "val_output-%d-%s_auc" % (ii, ln)
                obj += history.history[key][-1]
        else:
            val_auc = history.history['val_auc'][-1]
        obj = -val_auc / len(lname)
    epochs = len(history.history[list(history.history.keys())[0]])

    ml.info("# loss: "+str(obj))
    ml.timer("Iteration %d"%ml.nit, *cput0)
    ml.info("")

    if save_model:
        model.save(ml.filname_model)
        ml.info("Model saved to "+ml.filname_model)
        ml.model    =   model

    K.clear_session()

    return {'loss': np.float16(obj),
            'status': STATUS_OK,
            'epochs': epochs}


def optimize(X, y, lname,
             regression=True, hyperopt_step=100,
             arch=False, epochs=1000,
             X_val=False, y_val=False,
             model=False):
    '''
    Model hyperparameter optimization.

    Parameters
    ---
        X: np.array, features
        y: np.array, targets
        lname: str, name of the target property
        regression: boolean, whether it is a regression task
        hyperopt_step: int, number of steps for hyperopt
        arch: boolean or list, whether to use a fix architecture
        epochs: int, maximum epochs during the model training
        X_val: boolean or np.array, validation features
        y_val: boolean or np.array, validation targets
        model: boolean for keras.model, can start with a input model instead of building from scratch.
    '''
    np.random.seed(1234)
    if arch == False:
        architectures = [(128, 128),
                         (256, 256),
                         (512, 512),
                         (128, 128, 128),
                         (256, 256, 256),
                         (512, 512, 512)]
    else:
        architectures = [arch]
    bzs = [16, 32, 64, 128, 256, 512]
    ress = [True, False]
    bypasses = [True, False]
    if not model == False:
        space = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                 'batch_size': hp.choice('batch_size', bzs),
                 'beta_1': hp.uniform('beta_1', 0.75, 0.99),
                 'decay': hp.loguniform('decay', np.log(1e-5), np.log(1e-1)),
                 'amsgrad': True,
                 'patience': 50,
                 }
    else:
        space = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                 'drop_rate': hp.uniform('drop_rate', 0, 0.5),
                 'reg': hp.loguniform('reg', np.log(1e-5), np.log(5e-1)),
                 'batch_size': hp.choice('batch_size', bzs),
                 'hidden_size': hp.choice('hidden_size', architectures),
                 'beta_1': hp.uniform('beta_1', 0.75, 0.99),
                 'decay': hp.loguniform('decay', np.log(1e-5), np.log(1e-1)),
                 'res': hp.choice('res', ress),
                 'bypass': hp.choice('bypass', bypasses),
                 'amsgrad': True,
                 'patience': 50,
                 }
    objective_func = partial(train_model_hyperopt,
                             X=X,
                             y=y,
                             lname=lname,
                             regression=regression,
                             epochs=epochs,
                             X_val=X_val,
                             y_val=y_val,
                             input_model=model)
    trials = Trials()
    best_params = fmin(objective_func,
                       space,
                       algo=tpe.suggest,
                       trials=trials,
                       max_evals=hyperopt_step,
                       rstate=np.random.RandomState(0)
                       )
    if not model == False:
        best_params.update({'batch_size': bzs[best_params['batch_size']],
                            'amsgrad': True,
                            'patience': 10,
                            })
    else:
        best_params.update({'hidden_size': architectures[best_params['hidden_size']],
                            'batch_size': bzs[best_params['batch_size']],
                            'res': ress[best_params['res']],
                            'bypass': bypasses[best_params['bypass']],
                            'amsgrad': True,
                            'patience': 10,
                            })
    # One extra model training on train/validation set to get the number of epoch for the final model training.
    returned = train_model_hyperopt(best_params, X, y, lname,
                                    regression=regression, epochs=epochs,
                                    X_val=X_val, y_val=y_val,
                                    input_model=model)
    best_params.update({'epochs': returned['epochs']})
    return best_params
