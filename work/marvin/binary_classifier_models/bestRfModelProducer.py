
# coding: utf-8

# In[ ]:

# <api>
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.grid_search import GridSearchCV  # Perforing grid search

import sklearn.metrics as metrics
import work.marvin.binary_classifier_models.modelfit as modelfit


# In[ ]:

# <api>
# 强制要求最后一列是classfication label
def bestModelProducer(df, target, datamapper, fig_path):

    """
    # auto rf model generation, 3 steps:
    1. estimate optimal model parameters space for gridsearch,
    depends on sample size and feature size
    2. run gridsearch to find best parameter set
    3. train the best rf model using the best parameter set
    """
    traindf, testdf = modelfit.prepareDataforTraining(df, datamapper)
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]
    # estimate optimal parameters grid space
    param_grid = parameterGridInitialization(train)
    bestModel, accuracy, auc, cv_score = produceBestRFmodel(traindf, testdf, datamapper, param_grid, fig_path)
    return bestModel, traindf, testdf, accuracy, auc, cv_score


def initializationGridSearch(df, datamapper):
    traindf, testdf = modelfit.prepareDataforTraining(df, datamapper) 
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]
    # estimate optimal parameters grid space
    param_grid1, param_grid2 = parameterGridInitialization(train)
    return traindf, testdf, param_grid1, param_grid2


# ## Parameters Initialization 

# In[ ]:

# <api>
def max_feature_space(feature_size):
    fs_sqrt = math.sqrt(feature_size)
    if fs_sqrt > 10:
        max_feature = range(int(fs_sqrt-3), int(fs_sqrt*1.50), 2)
    else :
        max_feature = range(int(fs_sqrt), int(fs_sqrt*1.50), 2)
    return max_feature


# <api>
def n_estimators_space(train_size):
    """
     根据特征数量、样本数量初始化参数空间
    """
    if train_size > 2000:
        n_estimators_spc = range(50, 301, 20)
    else:
        n_estimators_spc = range(20, 100, 10)
    return n_estimators_spc


# <api>
def min_samples_leaf_space(train_size):
    """
     根据特征数量、样本数量初始化参数空间
    """
    min_samples_leaf_spc = [50, 100, 200, 500]
    return min_samples_leaf_spc


# <api>
def parameterGridInitialization(trainX):
    feature_size = trainX.shape[1] - 1
    train_size = trainX.shape[0]
    n_estimators_spc = n_estimators_space(train_size)
    min_samples_leaf_spc = min_samples_leaf_space(train_size)
    max_feature_spc = max_feature_space(feature_size)
    param_grid = {'n_estimators': n_estimators_spc,
                  'max_features': max_feature_spc,
                  'min_samples_leaf': min_samples_leaf_spc}
    return param_grid


# ## Best RF Model Producer 

# In[ ]:

# <api>
def rfGridSearch(train, labels_train, param_grid, seed=27):
    gsearch = GridSearchCV(estimator=RandomForestClassifier(oob_score=True, random_state=seed),
                           param_grid=param_grid, scoring='roc_auc',
                           n_jobs=-1, iid=False, cv=5)
    gsearch.fit(train, labels_train)
    best_parameters = gsearch.best_estimator_.get_params()
    best_n_estimators = best_parameters['n_estimators']
    best_max_features = best_parameters['max_features']
    best_min_samples_leaf = best_parameters['min_samples_leaf']
    return best_n_estimators, best_max_features, best_min_samples_leaf


# In[ ]:

# <api>
def produceBestRFmodel(traindf, testdf, datamapper, param_grid, fig_path=None, seed=27):
    # datamapper transform
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]            # 默认label为最后一列
    labels_train = train_array[:, -1]      # 默认label为最后一列
    test_array = datamapper.transform(testdf)
    test = test_array[:, :-1]
    labels_test = test_array[:, -1]

    # running grid search to get the best parameter set
    best_n_estimators, best_max_features, best_min_samples_leaf = rfGridSearch(train, labels_train, param_grid, seed=seed)

    rf_best = RandomForestClassifier(n_estimators=best_n_estimators,
                                     min_samples_leaf=best_min_samples_leaf,
                                     max_features=best_max_features,
                                     oob_score=True,
                                     random_state=seed)

    alg, train_predictions, train_predprob, cv_score = modelfit.modelfit(rf_best, datamapper,
                                                                         train, labels_train,
                                                                         test, labels_test,
                                                                         fig_path)

    accuracy = metrics.accuracy_score(labels_train, train_predictions)
    auc = metrics.roc_auc_score(labels_train, train_predprob)
    cv_score = [np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)]

    return alg, accuracy, auc, cv_score


# In[ ]:

def produceBestModel(traindf, testdf, datamapper, param_grid, fig_path=None, seed=27):
    return produceBestRFmodel(traindf, testdf, datamapper, param_grid, fig_path, seed)

