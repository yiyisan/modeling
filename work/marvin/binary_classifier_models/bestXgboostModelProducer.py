
# coding: utf-8

# In[ ]:

# <api>
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

from skopt.space import Categorical
import work.marvin.binary_classifier_models.modelfit as modelfit

import logging

logger = logging.getLogger(__name__)


# In[ ]:

# <api>
def bestModelProducer(data, target, datamapper, figpath):
    """
    auto xgboost model generation, 3 steps:
    1. estimate optimal model parameters space for gridsearch,
       depends on sample size and feature size
    2. run gridsearch to find best parameter set
    3. train the best GBDT model using the best parameter set
    """
    traindf, testdf = modelfit.prepareDataforTraining(data, datamapper)
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]

    # estimate optimal parameters grid space
    configspace = parameterGridInitialization(train)
    alg, accuracy, auc, cv_score = produceBestXgboostModel(traindf, testdf,
                                                     datamapper, configspace,
                                                     figpath)
    return alg, traindf, testdf, accuracy, auc, cv_score


# In[ ]:

# <api>
def parameterGridInitialization(trainX):
    feature_size = trainX.shape[1] - 1
    train_size = trainX.shape[0]

    n_estimators = [1000]

    subsample_spc = [0.6, 0.7, 0.8, 0.9]
    colsample_bytree_spc = [0.6, 0.7, 0.8, 0.9]

    gamma_spc = [i / 10.0 for i in range(0, 5)]
    reg_alpha_spc = [0, 0.001, 0.01, 0.1, 1, 10, 100]

    learning_rate_spc = [0.01, 0.05, 0.1]

    max_depth_spc = max_depth_space(feature_size)
    min_child_weight_spc = min_child_weight_space(train_size)

    # set learning_rate, run to get optiomal n_estimators
    param_grid1 = {'n_estimators': n_estimators}

    # most important parameters
    param_grid2 = {'max_depth': max_depth_spc,
                   'min_child_weight': min_child_weight_spc,
                   'subsample': subsample_spc,
                   'colsample_bytree': colsample_bytree_spc}

    # regularization parameters
    param_grid3 = {'gamma': gamma_spc, 'reg_alpha': reg_alpha_spc}

    # learning_rate parameters
    param_grid4 = {'learning_rate': learning_rate_spc}

    return param_grid1, param_grid2, param_grid3, param_grid4


# In[ ]:

# <api>
def configSpaceInitialization(trainX):      
    if trainX.shape[1] >= 10:
        skopt_grid = {'max_depth': (3, 9),
                      'learning_rate': (0.01, 0.1), 
                      'n_estimators': (50, 800), 
                      'objective': Categorical(('binary:logistic',)),
                      'gamma': (0, 3), 
                      'min_child_weight': (1, 5),
                      'subsample': (0.2, 0.9),
                      'colsample_bytree': (0.2, 0.9),
                      'reg_alpha': (1, 5),
                      'scale_pos_weight': (1, 5)}
    else:
        skopt_grid = {'max_depth': (2, 3),
                      'learning_rate': (0.01, 0.1), 
                      'n_estimators': (20, 100), 
                      'objective': Categorical(('binary:logistic',)),
                      'gamma': (0, 3),
                      'min_child_weight': (1, 5),
                      'subsample': (0.9, 1),
                      'colsample_bytree': (0.9, 1),
                      'reg_alpha': (1, 5),
                      'scale_pos_weight': (1, 5)}
    return skopt_grid


# In[ ]:

def searchBestParamsSkopt(train, labels_train, skopt_grid, search_alg, n_calls=100):
    experiment_setting = [(search_alg, skopt_grid, {'n_calls': n_calls})]
    experiment_result  = modelfit.run_experiments(experiment_setting, train, labels_train, XGBClassifier)
    test_accuracy = experiment_result[0]['Test accuracy']
    max_index = test_accuracy.index(max(test_accuracy))
    best_params = experiment_result[0]['Best parameters'][max_index]
    return best_params


# In[1]:

# <api>
def produceBestXgboostModel(traindf, testdf, datamapper,
                            param_grid1, param_grid2, param_grid3, param_grid4,
                            fig_path=None, seed=27):
    # datamapper transform
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]            # 默认label为最后一列
    labels_train = train_array[:, -1]      # 默认label为最后一列
    test_array = datamapper.transform(testdf)
    test = test_array[:, :-1]
    labels_test = test_array[:, -1]

    # running grid search to get the best parameter set 
    (best_subsample, best_estimators, best_learning_rate, best_max_depth,
    best_min_child_weight, best_colsample_bytree, best_gamma, best_reg_alpha) =\
    xgboostGridSearch(train, labels_train,
                      param_grid1, param_grid2, param_grid3, param_grid4)

    # train a gbm using the best parameter set
    xgboost_best = XGBClassifier(n_estimators=best_estimators, learning_rate=best_learning_rate,
                                 max_depth=best_max_depth, min_child_weight=best_min_child_weight,
                                 subsample=best_subsample, colsample_bytree=best_colsample_bytree,
                                 gamma=best_gamma, reg_alpha=best_reg_alpha,
                                 objective='binary:logistic', nthread=-1,
                                 scale_pos_weight=1, seed=seed)

    alg, train_predictions, train_predprob, cv_score = modelfit.modelfit(xgboost_best, datamapper,
                                                                         train, labels_train,
                                                                         test, labels_test,
                                                                         fig_path)

    accuracy = metrics.accuracy_score(labels_train, train_predictions)
    auc = metrics.roc_auc_score(labels_train, train_predprob)
    cv_score = [np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)]

    return alg, accuracy, auc, cv_score


# In[ ]:

# <api>
def max_depth_space(feature_size):
    if feature_size > 1000:
        max_depth = range(5, 14, 2)
    else:
        max_depth = range(3, 10, 2)
    return list(max_depth)


# <api>
def min_child_weight_space(train_size):
    if train_size > 10000:
        min_child_weight = range(3, 8, 1)
    else:
        min_child_weight = range(1, 6, 1)
    return list(min_child_weight)


# In[ ]:

# <api>
# train xgBoost to get best n_estimators
def xgBoostTrainBestn_estimators(alg, dtrain, dtest,
                                 useTrainCV=True, cv_folds=5,
                                 early_stopping_rounds=50):
    if not useTrainCV:
        return
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain, label=dtest)
    cvresult = xgb.cv(xgb_param, xgtrain,
                      num_boost_round=alg.get_params()['n_estimators'],
                      nfold=cv_folds,
                      metrics=['auc'],
                      early_stopping_rounds=early_stopping_rounds)
    return cvresult.shape[0]


# In[ ]:

# <api>
def xgboostGridSearch(train, labels_train,
                      param_grid1, param_grid2, param_grid3, param_grid4, seed=27):

    estimator = XGBClassifier(max_depth=3, min_child_weight=1,
                              gamma=0, subsample=0.8, learning_rate=0.1,
                              n_estimators=param_grid1['n_estimators'][0],
                              colsample_bytree=0.8,
                              objective='binary:logistic',
                              nthread=-1, scale_pos_weight=1, seed=27)

    best_estimators = xgBoostTrainBestn_estimators(estimator, train, labels_train)

    estimator1 = XGBClassifier(n_estimators=best_estimators,
                               learning_rate=0.1, gamma=0,
                               objective='binary:logistic',
                               nthread=-1, scale_pos_weight=1,
                               seed=seed)
    gsearch1 = GridSearchCV(estimator1,
                            param_grid=param_grid2, scoring='roc_auc',
                            n_jobs=1, iid=False, cv=5)
    gsearch1.fit(train, labels_train)

    best_parameters = gsearch1.best_estimator_.get_params()
    best_max_depth = best_parameters["max_depth"]
    best_min_child_weight = best_parameters['min_child_weight']
    best_subsample = best_parameters['subsample']
    best_colsample_bytree = best_parameters['colsample_bytree']
    estimator2 = XGBClassifier(n_estimators=best_estimators,
                               learning_rate=0.1,
                               max_depth=best_max_depth,
                               min_child_weight=best_min_child_weight,
                               subsample=best_subsample,
                               colsample_bytree=best_colsample_bytree,
                               objective='binary:logistic',
                               nthread=-1, scale_pos_weight=1,
                               seed=27)
    gsearch2 = GridSearchCV(estimator=estimator2,
                            param_grid=param_grid3, scoring='roc_auc',
                            n_jobs=1, iid=False, cv=5)
    gsearch2.fit(train, labels_train)

    best_parameters = gsearch2.best_estimator_.get_params()
    best_gamma = best_parameters["gamma"]
    best_reg_alpha = best_parameters["reg_alpha"]
   
    estimator3 = XGBClassifier(n_estimators=best_estimators,
                               max_depth=best_max_depth,
                               min_child_weight=best_min_child_weight,
                               subsample=best_subsample,
                               colsample_bytree=best_colsample_bytree,
                               gamma=best_gamma, reg_alpha=best_reg_alpha,
                               objective='binary:logistic',
                               nthread=-1, scale_pos_weight=1,
                               seed=seed),
    gsearch3 = GridSearchCV(estimator=estimator3,
                            param_grid=param_grid4, scoring='roc_auc',
                            n_jobs=1, iid=False, cv=5)
    gsearch3.fit(train, labels_train)

    best_parameters = gsearch3.best_estimator_.get_params()
    best_learning_rate = best_parameters["learning_rate"]

    estimator = XGBClassifier(n_estimators=param_grid1['n_estimators'][0]*2,
                              learning_rate=best_learning_rate,
                              max_depth=best_max_depth,
                              min_child_weight=best_min_child_weight,
                              subsample=best_subsample,
                              colsample_bytree=best_colsample_bytree,
                              gamma=best_gamma, reg_alpha=best_reg_alpha,
                              objective='binary:logistic', nthread=-1,
                              scale_pos_weight=1, seed=seed)
    best_estimators = xgBoostTrainBestn_estimators(estimator, train, labels_train)

    return (best_subsample, best_estimators, best_learning_rate, best_max_depth,
        best_min_child_weight, best_colsample_bytree, best_gamma, best_reg_alpha)


# In[ ]:

def produceBestModel(traindf, testdf, datamapper, param_grid, fig_path=None, seed=27):
    param_grid1, param_grid2, param_grid3, param_grid4 = param_grid
    return produceBestXgboostModel(traindf, testdf, datamapper,
                                   param_grid1, param_grid2, param_grid3, param_grid4,
                                   fig_path, seed)


# In[ ]:

# <api>
def optimizeBestModel(traindf, testdf, datamapper,
                      configspace, search_alg,
                      fig_path=None, n_calls=100, seed=27):
    # datamapper transform
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]            # 默认label为最后一列
    labels_train = train_array[:, -1]      # 默认label为最后一列
    test_array = datamapper.transform(testdf)
    test = test_array[:, :-1]
    labels_test = test_array[:, -1]

    # running grid search to get the best parameter set 
    best_params = searchBestParamsSkopt(train, labels_train, configspace, search_alg, n_calls)
    # train a gbm using the best parameter set
    xgboost_best = XGBClassifier(n_estimators=best_params['n_estimators'],
                                 learning_rate=best_params['learning_rate'],
                                 max_depth=best_params['max_depth'],
                                 min_child_weight=best_params['min_child_weight'],
                                 subsample=best_params['subsample'],
                                 colsample_bytree=best_params['colsample_bytree'],
                                 gamma=best_params['gamma'],
                                 reg_alpha=best_params['reg_alpha'],
                                 scale_pos_weight=best_params['scale_pos_weight'],
                                 objective='binary:logistic',
                                 nthread=-1, seed=seed)

    alg, train_predictions, train_predprob, cv_score = modelfit.modelfit(xgboost_best, datamapper,
                                                                         train, labels_train,
                                                                         test, labels_test,
                                                                         fig_path)

    accuracy = metrics.accuracy_score(labels_train, train_predictions)
    auc = metrics.roc_auc_score(labels_train, train_predprob)
    cv_score = [np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)]

    return alg, accuracy, auc, cv_score

