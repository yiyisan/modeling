
# coding: utf-8

# In[ ]:

# <api>
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

import work.marvin.binary_classifier_models.modelfit as modelfit


# In[ ]:

# <api>
def bestModelProducer(df, target, datamapper, fig_path):
    """
    # auto LR model generation, 3 steps:
    1. estimate optimal model parameters space for gridsearch, depends on sample size and feature size
    2. run gridsearch to find best parameter set
    3. train the best LR model using the best parameter set
    """
    traindf, testdf = modelfit.prepareDataforTraining(df, datamapper)

    param_grid = {'penalty':['l1','l2']} 
    
    bestModel, accuracy, auc, cv_score = produceBestLRmodel(traindf, testdf, datamapper, param_grid)
    return bestModel, traindf, testdf, accuracy, auc, cv_score


# In[ ]:

# <api>
def produceBestLRmodel(traindf, testdf, datamapper, param_grid, fig_path=None):
    # datamapper transform
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]            # 默认label为最后一列
    labels_train = train_array[:, -1]      # 默认label为最后一列
    test_array = datamapper.transform(testdf)
    test = test_array[:, :-1]
    labels_test = test_array[:, -1]
      
    # running grid search to get the best parameter set
    gsearch = GridSearchCV(estimator = LogisticRegression(random_state=10), param_grid = param_grid,
                           scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch.fit(train, labels_train)
    best_parameters = gsearch.best_estimator_.get_params()
    best_penalty = best_parameters['penalty']
    
    alg = LogisticRegression(penalty=best_penalty, random_state=10)   
    alg, train_predictions, train_predprob, cv_score = modelfit.modelfit(alg, datamapper, train, labels_train, test, labels_test, printFeatureImportance=False, fig_path=fig_path)
    
    accuracy = metrics.accuracy_score(labels_train, train_predictions)
    auc = metrics.roc_auc_score(labels_train, train_predprob)
    cv_score = [np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)]
    return alg, accuracy, auc, cv_score

