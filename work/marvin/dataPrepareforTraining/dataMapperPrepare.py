
# coding: utf-8

# In[1]:

# <api>
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain, OrdinalDomain
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import Imputer, Normalizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer

import base64
from six import string_types
from enum import Enum
import numpy as np
import pandas as pd
import logging

try:
    from exceptions import Exception
except:
    pass
logger = logging.getLogger(__name__)


# In[2]:

# <api>
class DataMapperError(Exception):
    pass


# In[ ]:

def ivt_missing(mis_val, col):
    if col not in mis_val:
        return 'as_is'
    if isinstance(mis_val[col], tuple) and 'mean' != mis_val[col][0]:
        return 'as_missing'
    elif 'drop row'.lower() == mis_val[col]:
        return 'return_invalid'
    else:
        raise Exception("""Invalid missing treatment
        of feature: {}""".format(col))


def onehot_encoder_with_missing(trn_series):
    unary = (trn_series.unique()[0] == 1)
    binary_with_na = is_binary(trn_series) and 'CreditX-NA' in set(trn_series)
    if unary or binary_with_na:
        return LabelEncoder()
    else:
        return LabelBinarizer()


def continuous_feature_transform(ftr_trf, col):
    prep = None
    if col in ftr_trf:
        if isinstance(ftr_trf[col], string_types):
            prep = FtrTransFunc(ftr_trf[col]).method
        else:
            prep = ftr_trf[col].method
    return prep


# In[3]:

class MisValFunc(Enum):
    AS_NEW_CLASS = 'as new class'
    DROP_ROW = 'drop row'
    MODE = 'mode'
    MEAN = 'mean'
    DEFAULT = 'default'

    def CategoricalTransform(self,
                             missing_value_replacement=None,
                             invalid_default='CreditX-NA',
                             invalid_value_treatment='as_missing'):
        if (self is MisValFunc.DEFAULT or self is MisValFunc.AS_NEW_CLASS):
            if missing_value_replacement is not None:
                missing_value_treatment = 'asValue'
            else:
                missing_value_replacement = invalid_default
        elif self is MisValFunc.MEAN or self is MisValFunc.DROP_ROW:
            missing_value_treatment = 'asMode'
        else:
            raise NotImplementedError
        return CategoricalDomain(invalid_value_treatment=invalid_value_treatment,
                                 invalid_default=invalid_default,
                                 missing_value_treatment=missing_value_treatment,
                                 missing_value_replacement=missing_value_replacement)

    def ContinuousTransform(self,
                            missing_value_replacement=None,
                            invalid_value_treatment='as_missing'):
        if self is MisValFunc.DEFAULT:
            missing_value_treatment = 'asValue'
        elif self is MisValFunc.DROP_ROW:
            missing_value_treatment = 'asMedian'
        elif self is MisValFunc.MEAN:
            missing_value_treatment = 'asMean'
        else:
            raise NotImplementedError
        return ContinuousDomain(invalid_value_treatment=invalid_value_treatment,
                                missing_value_treatment=missing_value_treatment,
                                missing_value_replacement=missing_value_replacement)

    def EncoderTransform(self, series):
        if self is MisValFunc.DEFAULT:
            encoder = onehot_encoder_with_missing(series)
        else:
            encoder = LabelEncoder() if is_binary(series) else LabelBinarizer()
        return encoder

    def ImputerTransform(self, series):
        if self is MisValFunc.DROP_ROW:
            return Imputer(strategy='median')
        elif self is MisValFunc.MODE:
            return Imputer(strategy='most_frequent')
        elif self is MisValFunc.MEAN:
            return Imputer(strategy='most_frequent')
        else:
            return None

    def apply(self, ftr, data, val=None):
        if self is MisValFunc.DEFAULT:
            set_default_value(data, ftr, val)
            return data
        elif self is MisValFunc.MEAN:
            set_mean(data, ftr)
            return data
        elif self is MisValFunc.DROP_ROW:
            drop_row(data, ftr)
            return data
        elif self is MisValFunc.AS_NEW_CLASS:
            newVal = set_as_new_class(data, ftr)
            return (self.value, newVal)
        else:
            raise NotImplementedError


# In[4]:

class FtrTransFunc(Enum):
    MIN_MAX_SCALER = 'MinMaxScaler'
    STANDARD_SCALER = 'StandardScaler'
    MAX_ABS_SCALER = 'MaxAbsScaler'
    NORMALIZER = 'Normalizer'
    BINARIZER = 'Binarizer'
    ONE_HOT_ENCODER = 'OneHotEncoder'
    NUMPY_LOG1P = 'NumPy.log1p'
    NUMPY_LOG = 'NumPy.log'

    @property
    def method(self):
        if self is FtrTransFunc.MIN_MAX_SCALER:
            return MinMaxScaler(copy=False)
        elif self is FtrTransFunc.STANDARD_SCALER:
            return StandardScaler(copy=False)
        elif self is FtrTransFunc.MAX_ABS_SCALER:
            return MaxAbsScaler(copy=False)
        elif self is FtrTransFunc.NORMALIZER:
            ft = FunctionTransformer(Normalizer(axis=0), False)
            ft.name = self.name
            return ft
        elif self is FtrTransFunc.BINARIZER:
            return LabelBinarizer(copy=False)
        elif self is FtrTransFunc.ONE_HOT_ENCODER:
            return OneHotEncoder()
        elif self is FtrTransFunc.NUMPY_LOG1P:
            ft = FunctionTransformer(np.log1p, False)
            ft.name = self.name
            return ft
        elif self is FtrTransFunc.NUMPY_LOG:
            ft = FunctionTransformer(np.log, False)
            ft.name = self.name
            return ft
        else:
            raise NotImplementedError

    def apply(self, series):
        after = self.method.fit_transform(series)
        if 'category' != series.dtype.name and contain_nan(after):
            raise Exception('feature contains nan when transformed by ' + self)
        if 'category' != series.dtype.name and contain_inf(after):
            raise Exception('feature contains inf when transformed by ' + self)
        return after


# In[5]:

# <api>
def b64_file_data(fig_path):
    fig_data = None
    with open(fig_path, 'r') as infile:
        fig_data = infile.read()
    return base64.b64encode(fig_data)


# In[6]:

# <api>
def drop_row(data, ftr):
    data.dropna(how='any', subset=[ftr], inplace=True)


def contain_nan(series):
    where = np.where(np.isnan(series))
    return 0 != len(where[0])


def contain_inf(series):
    where = np.where(np.isinf(series))
    return 0 != len(where[0])


# In[7]:

# <api>
def set_as_new_class(data, ftr):
    uniq_v = data[ftr].unique()
    uniq_v = uniq_v[~pd.isnull(uniq_v)]
    if 0 == len(uniq_v):
        raise Exception('all values of ' + ftr + ' are nan')

    # maybe need more check for the data type
    v = uniq_v[0]
    if isinstance(v, str):
        new_v = ftr + '_newclass'
    elif isinstance(v, (float, int)):
        new_v = uniq_v.astype('float32').max() + 1
        data[ftr] = data[ftr].astype('float32')
    else:
        raise Exception('categorical value is string or numerical?')
    set_default_value(data, ftr, new_v)
    data[ftr] = data[ftr].astype('category')
    return new_v


# In[8]:

# <api>
def set_default_value(data, ftr, v):
    if v is None:
        raise Exception('value is None')

    old_type = 'category'
    if 'category' == data[ftr].dtype.name:
        data[ftr] = data[ftr].astype('object', copy=True)
        v = str(v)
    else:
        data[ftr] = data[ftr].astype('float32', copy=True)
        v = float(v)
        old_type = 'float32'

    data[ftr].fillna(v, inplace=True)
    data[ftr] = data[ftr].astype(old_type, copy=True)


# In[9]:

# <api>
def set_mean(data, ftr):
    series = data[ftr]
    tmp = Imputer(axis=1).fit_transform(series)
    data[ftr].update(pd.Series(tmp[0]))


# In[10]:

# <api>
def deal_missing_value(data, mis_val):
    for f in mis_val.keys():
        m = mis_val[f]

        if MisValFunc.DROP_ROW == m:
            drop_row(data, f)
        elif MisValFunc.AS_NEW_CLASS == m:
            nc = set_as_new_class(data, f)
            mis_val[f] = (m, nc)
        elif MisValFunc.MEAN == m:
            set_mean(data, f)
        elif isinstance(m, tuple) and 'default' == m[0]:
            set_default_value(data, f, m[1])

    return mis_val


# In[11]:

# <api>
def move_target_last(data, target_col):
    reindex_col = [c for c in data.columns]
    if target_col not in reindex_col:
        return data
    reindex_col.remove(target_col)
    reindex_col.append(target_col)
    return data.reindex_axis(reindex_col, axis=1)


# In[12]:

# <api>
def is_binary(ftr_vlst):
    uniq_sp = ftr_vlst.unique()
    uniq_sp = uniq_sp[~pd.isnull(uniq_sp)]
    return 2 == len(uniq_sp)


# In[13]:

# <api>
def dataMapperBuilder(trn_d, categ_ftr, conti_ftr, invalid_ftr=None, mis_val=None, ftr_trf=None):
    """
    build dataFrameMapper according to colume type
    trn_d: traning data in DataFrame format
    categ_ftr: categorical feature(to_dummies)
    conti_ftr: continuous feature(feature transformer)
    mis_val: missing value treatment
    ftr_trf: feature transformer
    """
    invalid_ftr = invalid_ftr if invalid_ftr else []
    mis_val = mis_val if mis_val else {}
    ftr_trf = ftr_trf if ftr_trf else {}
    c_map = []
    for col in trn_d.columns:
        prep = []
        op_lst = []
        try:
            if col in categ_ftr:
                mis, val = mis_val.get(col, ('mean', None))
                misValFunc = MisValFunc(mis)
                mis_trans = misValFunc.EncoderTransform(trn_d[col])
                if mis_trans:
                    prep.append(mis_trans.fit(trn_d[col]))
                dom = misValFunc.CategoricalTransform(missing_value_replacement=val)
                op_lst.append(dom.fit(trn_d[col], name=col))
                op_lst.extend(prep)
            elif col in invalid_ftr:
                dom = OrdinalDomain(field_usage_treatment="supplementary")
                dom.fit(trn_d[col], name=col)
                op_lst.append(dom)
            elif col in conti_ftr:
                mis, val = mis_val.get(col, ('mean', None))
                misValFunc = MisValFunc(mis)
                # TODO: fix imputer impatible bug
                ftr_trans = continuous_feature_transform(ftr_trf, col)
                if ftr_trans:
                    prep.append(ftr_trans.fit(trn_d[col]))
                dom = misValFunc.ContinuousTransform(missing_value_replacement=val)
                op_lst.append(dom.fit(trn_d[col], name=col))
                op_lst.extend(prep)
            else:
                op_lst = None
        except Exception as e:
            logger.error(e)
        c_map.append(([col], op_lst))
    return DataFrameMapper(c_map)


# In[14]:

# <api>
def dataMapperPrepare(trn_d, parent_dfm, target_col=None):
    """
    datamapper prepare
    trn_d: train data
    parent_dfm: parent model datamapper
    target_col: specify target_col or target_col will be infered according to mapper
    """
    # none_ftr check
    none_ftr = [feature[0] for feature, mapper in parent_dfm.features
                if mapper is None]
    if target_col:
        data = move_target_last(trn_d, target_col)
    elif len(none_ftr) == 1 and not target_col:
        target_col = none_ftr[0]
        data = move_target_last(trn_d, target_col)
    else:
        raise Exception('df_mapper error')

    # domain ftr check
    invalid_ftr = [col for feature, mapper in parent_dfm.features
                   if mapper and mapper.domain_ == 'ordinaldomain'
                   for col in feature]
    categ_ftr = [col for feature, mapper in parent_dfm.features
                 if mapper and mapper.domain_ == 'categoricaldomain'
                 for col in feature]
    conti_ftr = [col for feature, mapper in parent_dfm.features
                 if mapper and mapper.domain_ == 'continuousdomain'
                 for col in feature]

    domain_ftr = set(invalid_ftr) | set(categ_ftr) | set(conti_ftr)
    if set(trn_d.columns) - set(none_ftr) - domain_ftr:
        raise DataMapperError("""
        datamapper consistency error:
        col = {}
        none = {}
        invalid = {}
        categorical = {}
        continuous = {}
        """.format(set(trn_d.columns), set(none_ftr),
                   set(invalid_ftr), set(categ_ftr), set(conti_ftr)))

    for ftr in categ_ftr:
        data[ftr] = data[ftr].str.encode('utf-8')

    # missing_value_check
    def isreal(mapper):
        return mapper.domain_ == 'categoricaldomain' or mapper.domain_ == 'continuousdomain'

    missing_value_treatment = [(name, mapper.missing_value_treatment)
                               for (name, mapper) in parent_dfm.features
                               if mapper and isreal(mapper)]
    missing_value_replacement = [(name, mapper.missingValueReplacement)
                                 for (name, mapper) in parent_dfm.features
                                 if mapper and isreal(mapper)]
    ftr_trf = {col: mapper.steps[-1][0] for (name, mapper) in parent_dfm.features
               for col in name if mapper and col in domain_ftr}
    mis_val = {}

    # missing value treatment
    for col_treatment, col_replacement in zip(missing_value_treatment, missing_value_replacement):
        name, treatment = col_treatment
        _, defaultVal = col_replacement
        if not isinstance(name, list):
            cols = [name]
        else:
            cols = name
        if treatment:
            for col in cols:
                data[col] = data[col].fillna(defaultVal)
                mis_val[col] = (treatment, defaultVal)

    dfm = dataMapperBuilder(data, categ_ftr, conti_ftr, invalid_ftr, mis_val, ftr_trf)
    return data, dfm, target_col


# In[15]:

# <api>
def buildTrainMapper(data, target, id_column=None):
    (transformed, categorical_features,
     continueous_features, invalid_feature) = prepare_for_training(data, target, id_column)
    datamapper = dataMapperBuilder(transformed, categorical_features, continueous_features)
    return transformed, datamapper


# In[16]:

# <api>
def prepare_for_training(data, target, id_column=None):
    """
    prepare_for_training shortcuts: using pandas infer
    data: train data
    target: label
    id_column: drop columns
    """
    transformed = data.copy()

    tmp = transformed.pop(target)
    transformed.insert(transformed.shape[1], target, tmp)

    if id_column and id_column in transformed.columns:
        invalid_features = transformed[id_column]
        transformed.drop(id_column, axis=1, inplace=True)

    contineous_describe = transformed.describe()
    non_features = set([target]) | set(invalid_features)
    continueous_features = set(contineous_describe.columns) - non_features
    categorical_features = set(transformed.columns) - set(continueous_features) - non_features
    for feature in categorical_features:
        transformed[feature] = transformed[feature].astype('category')
    for feature in continueous_features:
        transformed[feature] = transformed[feature].astype('float32')

    return transformed, categorical_features, continueous_features, invalid_features


# In[17]:

if __name__ == "__main__":
    a = pd.DataFrame([["1", 2, 3, 1], ["4", 5, 6, 1], ["7", 8, 9, 1]])
    dfm = dataMapperBuilder(a, [0], [1],
                            invalid_ftr=[2],
                            mis_val={0: ('asMode', "1")},
                            ftr_trf={1: 'log1p'})
    pd.DataFrame(dfm.features)
    tmp = pd.DataFrame(dfm.features)
    treat = tmp.ix[1, 1].steps[1][0]

    trn_d = pd.concat([a, pd.DataFrame([[np.nan, 11, 12]])], axis=0)
    assert dataMapperPrepare(trn_d, dfm)

