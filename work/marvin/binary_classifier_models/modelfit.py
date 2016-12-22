
# coding: utf-8

# In[1]:

# <api>
from time import time
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn import cross_validation  # Additional scklearn functions
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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


# In[2]:

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
        param_grid = self.parameterGrid(traindf, dfm)
        return self.model.produceBestModel(traindf, testdf, dfm, param_grid, fig_dir)

    def optimizeBestModel(self, traindf, testdf, dfm, fig_dir, search_alg='GP', n_calls=100):
        configspace = self.configspace(traindf, dfm)
        optimize_method = skopt_search(search_alg).search
        try:
            return self.model.optimizeBestModel(traindf, testdf, dfm, configspace,
                                                optimize_method, fig_dir, n_calls=n_calls)
        except Exception as e:
            logger.error("optimize {} with {} error: {}".format(self, search_alg, e))
            raise

    def parameterGrid(self, traindf, dfm):
        if self is BinaryClassifier.LR:
            return [{'penalty': ['l1', 'l2']}]
        else:
            train_array = dfm.transform(traindf)
            train = train_array[:, :-1]
            param_grid = self.model.parameterGridInitialization(train)
            return [param_grid] if self is BinaryClassifier.RF else param_grid

    def configspace(self, traindf, dfm):
        if self is BinaryClassifier.LR:
            return {'penalty': ['l1', 'l2']}
        else:
            train_array = dfm.transform(traindf)
            train = train_array[:, :-1]
            return self.model.configSpaceInitialization(train)


# In[14]:

# <api>
class skopt_search(Enum):
    gp = "GP"
    rf = "RF"
    gbrt = "GBRT"

    @property
    def method(self):
        if self is skopt_search.gp:
            from skopt import gp_minimize
            return gp_minimize
        elif self is skopt_search.rf:
            from skopt import forest_minimize
            return forest_minimize
        elif self is skopt_search.gbrt:
            from skopt import gbrt_minimize
            return gbrt_minimize

    def search(self, X_train, y_train, model_class, param_grid, loss, n_calls=100):
        """
        General method for applying `skopt_method` to the data.

        Parameters
        ----------
        X_train : np.array
            The design matrix, dimension `(n_samples, n_features)`.

        y_train : list or np.array
            The target, of dimension `n_samples`.

        model_class : classifier
            A classifier model in the mode of `sklearn`, with at least
            `fit` and `predict` methods operating on things like
            `X` and `y`.

        param_grid : dict
            Map from parameter names to pairs of values specifying the
            upper and lower ends of the space from which to sample.
            The values can also be directly specified as `skopt`
            objects like `Categorical`.

        loss : function or string
            An appropriate loss function or string recognizable by
            sklearn.cross_validation.cross_val_score. In sklearn, scores
            are positive and losses are negative because they maximize,
            but here we are minimizing so we always want smaller to mean
            better.

        n_calls : int
            Number of evaluations to do.

        Returns
        -------
        list of dict
            Each has keys 'loss' and 'params', where 'params' stores the
            values from `param_grid` for that run. The primary organizing
            value is 'loss'.
        Example
        -------
        >>> skopt_grid = {
                'max_depth': (4, 12),
                'learning_rate': (0.01, 0.5),
                'n_estimators': (20, 200),
                'objective' : Categorical(('multi:softprob',)),
                'gamma': (0, 0.5),
                'min_child_weight': (1, 5),
                'subsample': (0.1, 1),
                'colsample_bytree': (0.1, 1)}
        >>> res = skopt_search('RF').search(X, y, XGBClassifier, skopt_grid, LOG_LOSS, n_calls=10)

        To be followed by (see below):

        >>> best_params, best_loss = best_results(res)
        """
        logger.debug("---------------  skopt_search start --------------")
        param_keys, param_vecs = zip(*param_grid.items())
        param_keys = list(param_keys)
        param_vecs = list(param_vecs)

        def skopt_scorer(param_vec):
            params = dict(zip(param_keys, param_vec))
            logger.debug(params)
            err = cross_validated_scorer(
                X_train, y_train, model_class, params, loss)
            return err

        try:
            outcome = self.method(skopt_scorer, list(param_vecs), n_calls=n_calls)
            results = []
            for err, param_vec in zip(outcome.func_vals, outcome.x_iters):
                params = dict(zip(param_keys, param_vec))
                results.append({'loss': err, 'params': params})
            logger.debug("---------------  skopt_search end --------------")
        except Exception as e:
            logger.error(e)
            raise
        return results

        def skopt_scorer(param_vec):
            params = dict(zip(param_keys, param_vec))

            err = cross_validated_scorer(
                X_train, y_train, model_class, params, loss)
            return err

        outcome = self.method(skopt_scorer, list(param_vecs), n_calls=n_calls)
        results = []
        for err, param_vec in zip(outcome.func_vals, outcome.x_iters):
            params = dict(zip(param_keys, param_vec))
            results.append({'loss': err, 'params': params})
        logger.debug("---------------  skopt_search end --------------")
        return results


# In[19]:

class Loss(Enum):
    LOG_LOSS = 'neg_log_loss'


# In[20]:

# <api>
def prepareDataforTraining(transformed, datamapper, train_size=0.75):
    traindf, testdf = cross_validation.train_test_split(transformed, train_size=train_size)
    return traindf, testdf


# In[21]:

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
    try:
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
                sns.plt.savefig(fig_path)
    except Exception as e:
        raise Exception("save fig {} error: {}".format(fig_path, e))
    return alg, train_predictions, train_predprob, cv_score


# In[22]:

# <api>
def run_experiments(
        experimental_run,
        trainX,
        trainY,
        model_class,
        loss=Loss.LOG_LOSS,
        test_metric=roc_auc_score,
        random_state=None,
        dataset_name=None):
    """
    Basic experimental framework.

    Parameters
    ----------
    experimental_run : list of tuples
        These tuples should have exactly three members: the first one
        of `grid_search`, `randomized_search`, `hyperopt_search`,
        `skopt_gp_minimize`, `skopt_forest_minimize`, or
        `skopt_forest_gbrt`, the second an appropriate `param_grid`
        dict for that function, and the third a dict specifying
        keyword arguments to the search function.

    dataset : (np.array, iterable)
        A dataset (X, y) where `X` has dimension
        `(n_samples, n_features)` and `y` has
         dimension `n_samples`.

    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    loss : function or string
        An appropriate loss function or string recognizable by
        `sklearn.cross_validation.cross_val_score`. In `sklearn`, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    test_metric : function
        An `sklearn.metrics` function.

    random_state : int

    dataset_name : str or None
        Informal name to give the dataset. Purely for
        book-keeping.

    Returns
    -------
    list of dict
       Each dict is a results dictionary of the sort returned
       by `assess`.
    """
    X = trainX
    y = trainY

    skf = get_cross_validation_indices(
        X, y, n_folds=2, random_state=random_state)

    all_results = []
    # This loop can easily be parallelized, but doing so can
    # be tricky on some systems, since `cross_val_score`
    # calls `joblib` even if `n_jobs=1`, resulting in
    # nested parallel jobs even if there is no actual
    # parallelization elsewhere in the experimental run.
    for search_func, param_grid, kwargs in experimental_run:
        logger.info(search_func.__name__)
        all_results.append(
            assess(
                X, y,
                search_func=search_func,
                model_class=model_class,
                param_grid=param_grid,
                xval_indices=skf,
                loss=loss.value,
                test_metric=test_metric,
                dataset_name=dataset_name,
                search_func_args=kwargs))
        logger.info("--------------- assess end --------------")
    return all_results


# In[23]:

# <api>
def assess(
        X, y,
        search_func,
        model_class,
        param_grid,
        xval_indices,
        loss,
        test_metric=roc_auc_score,
        dataset_name=None,
        search_func_args={}):
    """
    The core of the experimental framework. This runs cross-validation
    and, for the inner loop, does cross-validation to find the optimal
    hyperparameters according to `search_func`. These optimal
    parameters are then used for an assessment in the outer
    cross-validation run.

    Parameters
    ----------
    X : np.array
        The design matrix, dimension `(n_samples, n_features)`.

    y : list or np.array
        The target, of dimension `n_samples`.

    search_func : function
        The search function to use. Can be `grid_search`,
        `randomized_search`, `hyperopt_search`, `skopt_gp_minimize`,
        `skopt_forest_minimize`, or `skopt_forest_gbrt`, all
        defined in this module. This choice has to be compatible with
        `param_grid`, in the sense that `grid_search` and
        `randomized_search` require a dict from strings to lists of
        values, `hyperopt_search` requires a dict from strings to
        hyperopt sampling functions, and the `skopt` functions
        require dicts from strings to (upper, lower) pairs of
        special `skopt` functions.

    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    param_grid : dict
        Map from parameter names to appropriate specifications of
        appropriate values for that parameter. This is not the
        expanded grid, but rather the simple map that can be expanded
        by `expand_grid` below (though not all methods call for that).
        This has to be compatible with  `search_func`, and all the
        values must be suitable arguments to `model_class` instances.

    loss : function or string
        An appropriate loss function or string recognizable by
        `sklearn.cross_validation.cross_val_score`. In `sklearn`, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    test_metric : function
        An `sklearn.metrics` function.

    xval_indices : list
        List of train and test indices into `X` and `y`. This defines
        the cross-validation. This is done outside of this method to
        allow for identical splits across different experiments.

    dataset_name : str or None
        Name for the dataset being analyzed. For book-keeping and
        display only.

    search_func_args : dict
        Keyword arguments to feed to `search_func`.

    Returns
    -------
    dict
        Accumulated information about the experiment:

        {'Test accuracy': list of float,
         'Cross-validation time':list of float,
         'Parameters sampled': list of int,
         'Method': search_func.__name__,
         'Model': model_class.__name__,
         'Dataset': dataset_name,
         'Best parameters': list of dict,
         'Mean test accuracy': float,
         'Mean cross-validation time': float,
         'Mean parameters sampled': float}   
    """
    logger.info("assess: {}".format(test_metric))
    data = {'Test accuracy': [],
            'Cross-validation time': [],
            'Parameters sampled': [],
            'Best parameters': [],
            'Method': search_func.__name__,
            'Model': model_class.__name__,
            'Dataset': dataset_name,
            'Best parameters': []}
    for cv_index, (train_index, test_index) in enumerate(xval_indices, start=1):
        logger.info("\t{}".format(cv_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        start = time()
        results = search_func(
            X_train,
            y_train,
            model_class,
            param_grid,
            loss,
            **search_func_args)
        data['Cross-validation time'].append(time() - start)
        data['Parameters sampled'].append(len(results))
        best_params = sorted(results, key=itemgetter('loss'), reverse=False)
        best_params = best_params[0]['params']
        data['Best parameters'].append(best_params)
        bestmod = model_class(**best_params)
        bestmod.fit(X_train, y_train)
        predictions = bestmod.predict(X_test)
        data['Test accuracy'].append(test_metric(y_test, predictions))
        data['Mean test accuracy'] = np.mean(data['Test accuracy'])
        data['Mean cross-validation time'] = np.mean(data['Cross-validation time'])
        data['Mean parameters sampled'] = np.mean(data['Parameters sampled'])

    return data


# In[24]:

# <api>
def get_cross_validation_indices(X, y, n_folds=5, random_state=None):
    """
    Use `StratifiedKFold` to create an `n_folds` cross-validator for
    the dataset defined by `X` and y`. Only `y` is used, but both are
    given for an intuitive interface; `X` could just as easily be used.
    """
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state)
    return skf.split(X, y)


# In[25]:

def cross_validated_scorer(
        X_train, y_train, model_class, params, loss, kfolds=5):
    """
    The scoring function used through this module, by all search
    functions.

    Parameters
    ----------
    X_train : np.array
        The design matrix, dimension `(n_samples, n_features)`.

    y_train : list or np.array
        The target, of dimension `n_samples`.

    model_class : classifier
        A classifier model in the mode of `sklearn`, with at least
        `fit` and `predict` methods operating on things like
        `X` and `y`.

    params : dict
        Map from parameter names to single appropriate values
        for that parameter. This will be used to build a model
        from `model_class`.

    loss : function or string
        An appropriate loss function or string recognizable by
        sklearn.cross_validation.cross_val_score. In sklearn, scores
        are positive and losses are negative because they maximize,
        but here we are minimizing so we always want smaller to mean
        better.

    kfolds : int
        Number of cross-validation runs to do.

    Returns
    -------
    float
       Average loss over the `kfolds` runs.
    """
    mod = model_class(**params)
    cv_score = -1 * cross_validation.cross_val_score(
        mod,
        X_train,
        y=y_train,
        scoring=loss,
        cv=kfolds,
        n_jobs=1).mean()
    return cv_score

