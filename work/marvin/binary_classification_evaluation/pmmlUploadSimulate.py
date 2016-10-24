
# coding: utf-8

# In[1]:

# <api>
import os
import pandas as pd
import requests
from IPython.display import display

try:
    from exceptions import Exception
except:
    pass

import logging
logger = logging.getLogger(__name__)


# In[2]:

# <api>
def upload_pmml(model, url="http://localhost:8080", name=None):
    headers = {"Content-Type": "application/xml"}
    try:
        with open(model) as fh:
            pmml = fh.read()
            if not name:
                pathname = os.path.basename(model)
                name = pathname.rstrip(".pmml")
            pmmlresult = requests.put("{}/openscoring/model/{}".format(url, name), data=pmml, headers=headers)
            return pmmlresult.json()
    except Exception as e:
        logger.error(e)
        raise Exception('post pmml failed')


# In[3]:

# <api>
def simulate_pmml(data, model, target, url="http://localhost:8080"):
    batchtest = data.to_dict()
    batchbody = [{"id": int(idx),
                  "arguments": {k: batchtest[k][idx].astype(object) 
                                if type(batchtest[k][idx]).__module__ == 'numpy' else batchtest[k][idx]
                                for k in batchtest.keys() if  k != target},
                  "target": batchtest[target][idx]} 
                 for idx in batchtest[target].keys()]
    testresult = []
    for testinst in batchbody:
        postbody = testinst.copy()
        y_true = postbody.pop("target")
        pmmlresult = requests.post("{}/openscoring/model/{}".format(url, model), json=postbody)
        ans = pmmlresult.json()
        testresult.append([ans['result'][target], y_true])
    return pd.DataFrame(testresult, columns=["predict", "true"])


# In[6]:

# <api>
def simulate_compare_pmml(data, model, target,test_predprob, url="http://localhost:8080"):
    batchtest = data.to_dict()
    batchbody = [{"id": int(idx),
                  "arguments": {k: batchtest[k][idx].astype(object)
                                if type(batchtest[k][idx]).__module__ == 'numpy' else batchtest[k][idx]
                                for k in batchtest.keys() if  k != target},
                  "target": batchtest[target][idx]} 
                 for idx in batchtest[target].keys()]
    prob_pmml = {}
    for testinst in batchbody:
        postbody = testinst.copy()
        y_true = postbody.pop("target")
        pmmlresult = requests.post("{}/openscoring/model/{}".format(url, model), json=postbody)
        ans = pmmlresult.json()
        prob_pmml.update({postbody['id']: ans['result']['probability_1']})

    prob_alg = {data.index[i]: test_predprob[i] for i in range(len(test_predprob))}
    count_compare,compare_detail = probability_statistics(prob_pmml, prob_alg, data)
    logger.info('--- Consistency Compare ---')
    display(count_compare)
    logger.info('--- Inconsistency Detail ---')
    display(compare_detail)
    return count_compare, compare_detail


# In[5]:

# <api>
def probability_statistics(prob_pmml,prob_alg,data):
    
    prob_pmml_approx = {key: round(prob_pmml[key], 4) for key in prob_pmml}
    prob_alg_approx = {key: round(prob_alg[key], 4) for key in prob_alg}

    num_consist = 0
    index_id_list = []
    local_prob_list = []
    server_prob_list = []

    for key in prob_pmml_approx:
        if prob_pmml_approx[key] == prob_alg_approx[key]:
            num_consist = num_consist + 1
        else:
            index_id_list.append(key)
            local_prob_list.append(prob_alg_approx[key])
            server_prob_list.append(prob_pmml_approx[key])
    num_inconsist = len(prob_pmml_approx) - num_consist
    count_compare = pd.DataFrame({'ConsistentNumber': [num_consist], 'InconsistentNumber': [num_inconsist]})
    compare_detail = pd.DataFrame({'IndexID': index_id_list, 'LocalProbability': local_prob_list,
                                   'ServerProbability': server_prob_list})
    compare_detail = pd.merge(compare_detail, data,
                              how='inner', left_on=['IndexID'], right_index=True)

    return count_compare, compare_detail

