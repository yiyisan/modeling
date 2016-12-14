
# coding: utf-8

# In[ ]:

# <api>
import math
import numpy as np
from sklearn import metrics  # Additional scklearn functions
from sklearn.ensemble import GradientBoostingClassifier  # GBM modelorithm
from sklearn.grid_search import GridSearchCV  # Perforing grid search
import work.marvin.binary_classifier_models.modelfit as modelfit


# In[ ]:

# <api>
def bestModelProducer(df, target, datamapper, fig_path):
    """
    # auto GBDT model generation, 3 steps:
    1. estimate optimal model parameters space for gridsearch,
    depends on sample size and feature size
    2. run gridsearch to find best parameter set
    3. train the best GBDT model using the best parameter set
    """
    traindf, testdf = modelfit.prepareDataforTraining(df, datamapper)
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]
    # estimate optimal parameters grid space
    param_grid1, param_grid2 = parameterGridInitialization(train)
    bestModel, accuracy, auc, cv_score = produceBestGBMmodel(traindf, testdf, datamapper,
                                                             param_grid1, param_grid2, fig_path)
    return bestModel, traindf, testdf, accuracy, auc, cv_score


def initializationGridSearch(df, datamapper):
    """
     根据特征数量、样本数量初始化GBM参数空间
    """
    traindf, testdf = modelfit.prepareDataforTraining(df, datamapper)
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]
    # estimate optimal parameters grid space
    param_grid1, param_grid2 = parameterGridInitialization(train)


# In[8]:

# <api>
def produceBestGBMmodel(traindf, testdf, datamapper,
                        param_grid1, param_grid2,
                        fig_path=None, seed=27):
    # datamapper transform
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]
    labels_train = train_array[:, -1]

    test_array = datamapper.transform(testdf)
    test = test_array[:, :-1]
    labels_test = test_array[:, -1]

    # running grid search to get the best parameter set  
    best_subsample, best_estimators, best_learning_rate, best_max_depth, best_max_feature, best_min_samples_split = gbmGridSearch(train,
                                                                                                                                  labels_train,
                                                                                                                                  param_grid1,
                                                                                                                                  param_grid2)

    # train a gbm using the best parameter set
    gbm_best = GradientBoostingClassifier(learning_rate=best_learning_rate,
                                          n_estimators=best_estimators,
                                          max_depth=best_max_depth,
                                          min_samples_split=best_min_samples_split,
                                          subsample=best_subsample,
                                          max_features=best_max_feature,
                                          random_state=seed)

    alg, train_predictions, train_predprob, cv_score = modelfit.modelfit(gbm_best, datamapper,
                                                                         train, labels_train,
                                                                         test, labels_test,
                                                                         fig_path=fig_path)

    accuracy = metrics.accuracy_score(labels_train, train_predictions)
    auc = metrics.roc_auc_score(labels_train, train_predprob)
    cv_score = [np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)]

    return alg, accuracy, auc, cv_score


# In[ ]:

def produceBestModel(traindf, testdf, datamapper, param_grid, fig_path=None, seed=27):
    param_grid1, param_grid2 = param_grid
    return produceBestGBMmodel(traindf, testdf, datamapper,
                               param_grid1, param_grid2,
                               fig_path, seed)


# In[9]:

# <api>
def n_estimators_space(train_size):
    if train_size > 10000:
        n_estimators_spc = range(200, 1001, 200)
    else:
        n_estimators_spc = range(50, 201, 50)
    return list(n_estimators_spc)


def min_samples_split_space(train_size):
    return list(range(min(train_size, 100), min(train_size, 601), 100))


def max_feature_space(feature_size):
    fs_sqrt = math.sqrt(feature_size)
    if fs_sqrt > 10:
        max_feature = range(int(fs_sqrt - 3), int(fs_sqrt * 1.50), 2)
    else:
        max_feature = range(int(fs_sqrt), int(fs_sqrt * 1.50), 2)    
    return list(max_feature)


def max_depth_space(feature_size):
    return [3, 5, 7, 9]


def parameterGridInitialization(trainX):
    feature_size = trainX.shape[1] - 1
    train_size = trainX.shape[0]

    subsample_spc = [0.6, 0.7, 0.8, 0.9]
    learning_rate_spc = [0.01, 0.05, 0.1]
    n_estimators_spc = n_estimators_space(train_size)
    min_samples_split_spc = min_samples_split_space(train_size)
    max_feature_spc = max_feature_space(feature_size)
    max_depth_spc = max_depth_space(feature_size)
    min_samples_split_spc = min_samples_split_space(train_size)

    # most important parameters
    param_grid1 = {'subsample': subsample_spc, 'n_estimators': n_estimators_spc,
                   'learning_rate': learning_rate_spc}
    # tree specific parameters
    param_grid2 = {'max_depth': max_depth_spc, 'max_features': max_feature_spc, 
                   'min_samples_split': min_samples_split_spc}

    return param_grid1, param_grid2


# In[10]:

# <api>
def gbmGridSearch(train, labels_train, param_grid1, param_grid2, seed=27):
    gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(min_samples_split=30,
                                                                 max_features='sqrt',
                                                                 max_depth=5,
                                                                 random_state=10),
                            param_grid=param_grid1, scoring='roc_auc',
                            n_jobs=-1, pre_dispatch='2*n_jobs', iid=False, cv=5)
    gsearch1.fit(train, labels_train)

    best_parameters = gsearch1.best_estimator_.get_params()
    best_subsample = best_parameters["subsample"]
    best_estimators = best_parameters['n_estimators']
    best_learning_rate = best_parameters['learning_rate']

    gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(subsample=best_subsample,
                                                                   n_estimators=best_estimators,
                                                                   learning_rate=best_learning_rate,
                                                                   random_state=seed),
                            param_grid=param_grid2, scoring='roc_auc',
                            n_jobs=-1, pre_dispatch='2*n_jobs', iid=False, cv=5)
    gsearch2.fit(train, labels_train)

    best_parameters2 = gsearch2.best_estimator_.get_params()
    best_max_depth = best_parameters2["max_depth"]
    best_max_feature = best_parameters2["max_features"]
    best_min_samples_split = best_parameters2["min_samples_split"]

    return best_subsample, best_estimators, best_learning_rate, best_max_depth, best_max_feature, best_min_samples_split

