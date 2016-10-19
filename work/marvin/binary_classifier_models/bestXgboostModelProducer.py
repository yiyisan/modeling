
# coding: utf-8

# In[ ]:

# <api>
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

import logging
import work.marvin.binary_classifier_models.modelfit as modelfit


logger = logging.getLogger(__name__)


# In[ ]:

# <api>
def bestModelProducer(data, target, datamapper, figpath):
    """
    # auto xgboost model generation, 3 steps:
    1. estimate optimal model parameters space for gridsearch, depends on sample size and feature size
    2. run gridsearch to find best parameter set
    3. train the best GBDT model using the best parameter set
    """
    traindf, testdf = modelfit.prepareDataforTraining(data, datamapper)
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1] 
    
    # estimate optimal parameters grid space
    param_grid1, param_grid2, param_grid3, param_grid4 = parameterGridInitialization(train)
    alg, accuracy, auc, cv_score = produceBestXgboostModel(traindf, testdf, datamapper,
                                                           param_grid1, param_grid2, param_grid3, param_grid4,
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
    param_grid2 = {'max_depth': max_depth_spc, 'min_child_weight': min_child_weight_spc, 
                   'subsample': subsample_spc, 'colsample_bytree': colsample_bytree_spc}
    
    # regularization parameters
    param_grid3 = {'gamma': gamma_spc, 'reg_alpha': reg_alpha_spc}
    
    # learning_rate parameters
    param_grid4 = {'learning_rate': learning_rate_spc}
    
    return param_grid1, param_grid2, param_grid3, param_grid4


# In[ ]:

# <api>
def produceBestXgboostModel(traindf, testdf, datamapper, param_grid1, param_grid2, param_grid3, param_grid4, fig_path=None):

    # datamapper transform
    train_array = datamapper.transform(traindf)
    train = train_array[:, :-1]            # 默认label为最后一列
    labels_train = train_array[:, -1]      # 默认label为最后一列
    test_array = datamapper.transform(testdf)
    test = test_array[:, :-1]
    labels_test = test_array[:, -1]
 
    # running grid search to get the best parameter set  
    best_subsample,best_estimators,best_learning_rate,best_max_depth,best_min_child_weight,best_colsample_bytree,best_gamma,best_reg_alpha = xgboostGridSearch(train, labels_train, param_grid1, param_grid2, param_grid3, param_grid4)
    
    # train a gbm using the best parameter set
    xgboost_best = XGBClassifier(n_estimators=best_estimators, learning_rate=best_learning_rate,
                                 max_depth=best_max_depth, min_child_weight=best_min_child_weight,
                                 subsample=best_subsample, colsample_bytree=best_colsample_bytree, 
                                 gamma=best_gamma, reg_alpha=best_reg_alpha, objective='binary:logistic', nthread=4,
                                 scale_pos_weight=1, seed=27)

    alg, train_predictions, train_predprob, cv_score = modelfit.modelfit(xgboost_best, datamapper, train, labels_train, test, labels_test, fig_path)

    accuracy = metrics.accuracy_score(labels_train, train_predictions)
    auc = metrics.roc_auc_score(labels_train, train_predprob)
    cv_score = [np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)]

    return alg, accuracy, auc, cv_score


# In[ ]:

# <api>
def _modelfit(alg, datamapper, train, labels_train, test, labels_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train, label=labels_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics=['auc'], early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train, labels_train, eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(train)
    dtrain_predprob = alg.predict_proba(train)[:,1]

    #Print Feature Importance:
    trans = [["{}={}".format(name, str(cls)) for cls in mapper.classes_.tolist()]
             if isinstance(mapper, LabelBinarizer) else [name] for (name, mapper) in datamapper.features]
    feature_indices = [i for sublist in trans for i in sublist][1:]
    xg_feature_importances = pd.DataFrame([alg.feature_importances_], columns = feature_indices)
    sorted_feature_importances = xg_feature_importances.ix[0, :].sort_values(ascending=False).index[:20] # 这里的20表示按重要性顺序取前20个
    feature_importances = xg_feature_importances[sorted_feature_importances]
        
    sns.barplot(x=feature_importances.columns, y=np.array(feature_importances)[0,:])
    sns.plt.title('Feature Importances')
    sns.plt.xlabel('Feature')
    sns.plt.xticks(rotation=90)
    sns.plt.ylabel('Feature Importance Score')
    sns.plt.savefig('featureimportance.png')
        
    #Print model report:
    logger.info("Model Report")
    logger.info("Accuracy : %.4g" % metrics.accuracy_score(labels_train, dtrain_predictions))
    logger.info("AUC Score (Train): %f" % metrics.roc_auc_score(labels_train, dtrain_predprob))

    return alg


# In[ ]:

# <api>
def max_depth_space(feature_size):
    if feature_size > 1000 :
        max_depth = range(5,14,2)
    else :
        max_depth = range(3,10,2)      
    return max_depth

# <api>
def min_child_weight_space(train_size):
    if train_size > 10000 :
        min_child_weight = range(3,8,1)
    else :
        min_child_weight = range(1,6,1)      
    return min_child_weight


# In[ ]:

# <api>
# train xgBoost to get best n_estimators
def xgBoostTrainBestn_estimators(alg, dtrain, dtest, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=dtest)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics=['auc'], early_stopping_rounds=early_stopping_rounds)
        return cvresult.shape[0]


# In[ ]:

# <api>
def xgboostGridSearch(train, labels_train, param_grid1, param_grid2, param_grid3, param_grid4):
    
    estimator = XGBClassifier(max_depth=3, min_child_weight=1, gamma=0, subsample=0.8, learning_rate=0.1,
                              n_estimators=param_grid1['n_estimators'][0], colsample_bytree=0.8, objective='binary:logistic', 
                              nthread=4, scale_pos_weight=1, seed=27)
    
    best_estimators = xgBoostTrainBestn_estimators(estimator, train, labels_train)
    
    gsearch1 = GridSearchCV(estimator = XGBClassifier(n_estimators=best_estimators, learning_rate=0.1,
                                                      gamma=0, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, 
                                                      seed=27), param_grid = param_grid2, scoring='roc_auc',n_jobs=4,
                                                      iid=False, cv=5)  
    gsearch1.fit(train, labels_train)
    
    best_parameters = gsearch1.best_estimator_.get_params()
    best_max_depth = best_parameters["max_depth"]   
    best_min_child_weight = best_parameters['min_child_weight']
    best_subsample = best_parameters['subsample']
    best_colsample_bytree = best_parameters['colsample_bytree']
    
    gsearch2 = GridSearchCV(estimator = XGBClassifier(n_estimators=best_estimators, learning_rate=0.1,
                                                      max_depth=best_max_depth, min_child_weight=best_min_child_weight,
                                                      subsample=best_subsample, colsample_bytree=best_colsample_bytree,
                                                      objective= 'binary:logistic', nthread=4, scale_pos_weight=1, 
                                                      seed=27), param_grid = param_grid3, scoring='roc_auc', n_jobs=4,
                                                      iid=False, cv=5)  
    gsearch2.fit(train, labels_train)  
    
    best_parameters = gsearch2.best_estimator_.get_params()    
    best_gamma = best_parameters["gamma"]
    best_reg_alpha = best_parameters["reg_alpha"]
    
    gsearch3 = GridSearchCV(estimator = XGBClassifier(n_estimators=best_estimators, 
                                                      max_depth=best_max_depth, min_child_weight=best_min_child_weight,
                                                      subsample=best_subsample, colsample_bytree=best_colsample_bytree,
                                                      gamma=best_gamma, reg_alpha=best_reg_alpha,
                                                      objective= 'binary:logistic', nthread=4, scale_pos_weight=1, 
                                                      seed=27), param_grid = param_grid4, scoring='roc_auc',n_jobs=4,
                                                      iid=False, cv=5)  
    gsearch3.fit(train, labels_train)  
    
    best_parameters = gsearch3.best_estimator_.get_params()    
    best_learning_rate = best_parameters["learning_rate"]
    
    estimator = XGBClassifier(n_estimators=param_grid1['n_estimators'][0]*2, learning_rate=best_learning_rate,
                              max_depth=best_max_depth, min_child_weight=best_min_child_weight,
                              subsample=best_subsample, colsample_bytree=best_colsample_bytree, 
                              gamma=best_gamma, reg_alpha=best_reg_alpha, objective='binary:logistic', nthread=4, 
                              scale_pos_weight=1, seed=27)
    
    best_estimators = xgBoostTrainBestn_estimators(estimator, train, labels_train)
    
    return best_subsample, best_estimators, best_learning_rate, best_max_depth, best_min_child_weight, best_colsample_bytree, best_gamma, best_reg_alpha

