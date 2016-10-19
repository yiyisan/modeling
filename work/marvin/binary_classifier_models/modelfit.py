
# coding: utf-8

# In[1]:

# <api>
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation #Additional scklearn functions

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = 8, 4


# In[1]:

# <api>
def prepareDataforTraining(transformed, datamapper, train_size=0.75):
    traindf, testdf = sklearn.cross_validation.train_test_split(transformed, train_size=train_size)
    return traindf, testdf


# In[ ]:

# <api>
def modelfit(alg, datamapper, train, labels_train, test, labels_test, fig_path=None, performCV=True, printFeatureImportance=True, cv_folds=5, most_importance_n=20):
    alg.fit(train, labels_train)
    train_predictions = alg.predict(train)
    train_predprob = alg.predict_proba(train)[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, train, labels_train, cv=cv_folds, scoring='roc_auc')
   
    #Print Feature Importance:
    if printFeatureImportance:
        #print(datamapper)
        feature_list = [mapper.data_ for (name, mapper) in datamapper.features if mapper]
        feature_indices = [feature for sublist in feature_list for feature in sublist]
        gbtree_feature_importances = pd.DataFrame([alg.feature_importances_], columns = feature_indices)
        sorted_feature_importances = gbtree_feature_importances.ix[0, :].sort_values(ascending=False).index[:most_importance_n]
        feature_importances = gbtree_feature_importances[sorted_feature_importances]
        # Plot barchart
        sns.plt.clf()
        sns.plt.figure(figsize=(8, 6))
        sns.barplot(x=feature_importances.columns, y=np.array(feature_importances)[0,:], label='small')
        sns.plt.title('Feature Importances')
        sns.plt.xlabel('Feature')
        sns.plt.xticks(rotation=90)
        sns.plt.ylabel('Feature Importance Score')
        sns.plt.tight_layout()
        if fig_path is not None:
            sns.plt.savefig(fig_path)

 
    return alg, train_predictions, train_predprob, cv_score

