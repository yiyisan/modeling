
# coding: utf-8

# In[1]:

# <api>
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation  # Additional scklearn functions
from enum import Enum
import matplotlib
import seaborn as sns

try:
    from exceptions import Exception
except:
    pass

import logging

logger = logging.getLogger(__name__)

matplotlib.use('agg')


# In[ ]:

class BinaryClassifier(Enum):
    GBM = 'GBM'
    XGB = 'XGBOOST'
    LGB = 'LightGBM'
    RF = 'RF'
    LR = 'LR'

    @property
    def model(self):
        if self is BinaryClassifier.GBM:
            import work.marvin.binary_classifier_models.bestGbdtModelProducer
            return work.marvin.binary_classifier_models.bestGbdtModelProducer
        elif self is BinaryClassifier.XGB:
            import work.marvin.binary_classifier_models.bestXgboostModelProducer
            return work.marvin.binary_classifier_models.bestXgboostModelProducer
        elif self is BinaryClassifier.RF:
            import work.marvin.binary_classifier_models.bestRfModelProducer
            return work.marvin.binary_classifier_models.bestRfModelProducer
        elif self is BinaryClassifier.LR:
            import work.marvin.binary_classifier_models.bestLrModelProducer
            return work.marvin.binary_classifier_models.bestLrModelProducer

    def produceBestModel(self, traindf, testdf, dfm, fig_dir):
        param_grid = self.configspace(traindf, dfm)
        logger.debug(param_grid)
        return self.model.produceBestModel(traindf, testdf, dfm, param_grid, fig_dir)

    def configspace(self, traindf, dfm):
        if self is BinaryClassifier.LR:
            return [{'penalty': ['l1', 'l2']}]
        else:
            train_array = dfm.transform(traindf)
            train = train_array[:, :-1]
            param_grid = self.model.parameterGridInitialization(train)
            return [param_grid] if self is BinaryClassifier.RF else param_grid


# In[1]:

# <api>
def prepareDataforTraining(transformed, datamapper, train_size=0.75):
    traindf, testdf = sklearn.cross_validation.train_test_split(transformed, train_size=train_size)
    return traindf, testdf


# In[1]:

# <api>
def modelfit(alg, datamapper, train, labels_train, test, labels_test,
             fig_path=None, cv_folds=5, most_importance_n=20):
    alg.fit(train, labels_train)
    train_predictions = alg.predict(train)
    train_predprob = alg.predict_proba(train)[:, 1]

    cv_score = cross_validation.cross_val_score(alg, train, labels_train,
                                                cv=cv_folds, n_jobs=cv_folds, scoring='roc_auc')

    feature_list = [mapper.data_ for (name, mapper) in datamapper.features if mapper]
    feature_indices = [feature for sublist in feature_list for feature in sublist]
    if hasattr(alg, 'feature_importances_'):
        feature_importances = pd.DataFrame([alg.feature_importances_], columns=feature_indices)
    elif hasattr(alg, 'coef_'):
        feature_importances = pd.DataFrame(alg.coef_, columns=feature_indices)
    else:
        raise Exception('unrecognized algorithm')
    sorted_abs_importances = feature_importances.ix[0, :].abs().sort_values(ascending=False)
    sorted_feature_importances = sorted_abs_importances.index[:most_importance_n]
    feature_importances = feature_importances[sorted_feature_importances]
    logger.debug("plot feature_importances")
    # Plot barchart
    sns.plt.clf()
    sns.plt.figure(figsize=(8, 6))
    sns.barplot(x=[col.decode("utf-8") for col in feature_importances.columns],
                y=np.array(feature_importances)[0, :],
                label='small')
    sns.plt.title('Feature Importances')
    sns.plt.xlabel('Feature')
    sns.plt.xticks(rotation=90)
    sns.plt.ylabel('Feature Importance Score')
    sns.plt.tight_layout()
    if fig_path is not None:
        try:
            sns.plt.savefig(fig_path)
        except Exception as e:
            raise Exception("save fig {} error".format(fig_path))
    return alg, train_predictions, train_predprob, cv_score


# In[ ]:



