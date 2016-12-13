
# coding: utf-8

# In[1]:

# <api>
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt 


# In[2]:

# <api>
def algEvaluateOnTestSet(alg, testdf, datamapper, ks_fig_path, sub_fig_path):

    test_array = datamapper.transform(testdf)
    test = test_array[:, :-1]
    labels_test = test_array[:, -1]

    test_predprob = applyAlgOnTestSet(alg, test)

    # plots: KS, ROC, precision_recall, precision_cutoff, recall_cutoff
    ks_val, ks_x, p, r = ks_curve(labels_test, test_predprob, ks_fig_path)

    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    roc_curve_plot(labels_test, test_predprob)
    plt.subplot(2, 2, 2)
    precision_recall_curve_plot(labels_test, test_predprob)
    plt.subplot(2, 2, 3)
    precision_cutoff_curve(labels_test, test_predprob)
    plt.subplot(2, 2, 4)
    recall_cutoff_curve(labels_test, test_predprob)
    plt.tight_layout()
    plt.savefig(sub_fig_path)

    return test_predprob, ks_val, ks_x, p, r


# In[3]:

# <api>
def applyAlgOnTestSet(alg, test):
    test_predprob = alg.predict_proba(test)[:, 1]
    return test_predprob


# In[4]:

# <api>
def greaterThan(a, b):
    return 1 if a > b else 0


# <api>
def metricsPlot(x, y, xlab, ylab, title):
    plt.plot(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc='lower right')
    #plt.show()


# <api>
def prec(Y_true, Y_predprob, t): 
    vfunc = np.vectorize(greaterThan) 
    return metrics.precision_score(Y_true, vfunc(Y_predprob, t))


# <api>
def rec(Y_true, Y_predprob, t): 
    vfunc = np.vectorize(greaterThan) 
    return metrics.recall_score(Y_true, vfunc(Y_predprob, t))


# In[5]:

# <api>
def roc_curve_plot(Y_true,Y_predprob):
    from sklearn.metrics import roc_curve
    fpr,tpr,thresh = roc_curve(Y_true, Y_predprob)
    auc = round(auc_calculate(Y_true, Y_predprob), 4)
    plt.plot(fpr, tpr, label="ROC-AUC overall,\n AUC Score={}".format(auc))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot([0, 1], [0, 1])
    plt.title("ROC Curve")
    plt.legend(loc="lower right")


def precision_recall_curve_plot(Y_true, Y_predprob):
    precision,recall,threshold = precision_recall_curve(Y_true, Y_predprob)
    metricsPlot(precision, recall,
                "Precision", "Recall", "Precision Vs Recall Curve")


def precision_cutoff_curve(Y_true, Y_predprob):
    vfunc = np.vectorize(greaterThan)
    max_t = max(Y_predprob)
    t = []
    prec = []
    for i in range(0, 101, 1):
        if i / 100.0000 <= max_t:
            t.append(i / 100.0000)
            prec.append(metrics.precision_score(Y_true, vfunc(Y_predprob, t[i])))
    metricsPlot(t, prec, "cut-off", "Precision", "Precision Vs Cut-off Curve")

     
def recall_cutoff_curve(Y_true, Y_predprob):
    vfunc = np.vectorize(greaterThan) 
    max_t = max(Y_predprob)
    t = []
    rec = []
    for i in range(0,101,1):
        if i / 100.0000 <= max_t:
            t.append(i / 100.000)
            rec.append(metrics.recall_score(Y_true, vfunc(Y_predprob, t[i])))
    metricsPlot(t, rec, "cut-off", "Recall", "Recall Vs Cut-off Curve")


def auc_calculate(Y_true, Y_predprob):
    fpr,tpr,thresh = roc_curve(Y_true,Y_predprob)
    return round(metrics.auc(fpr, tpr),4)


# In[4]:

# <api>
def ks_curve(Y_true, Y_predprob, fig_path):
    # Kolmogorov-Smirnov Test
    df = pd.DataFrame({'Y_truth': Y_true, 'Y_predprob_1': Y_predprob})
    a = df[df.Y_truth == 0]['Y_predprob_1']
    b = df[df.Y_truth == 1]['Y_predprob_1']

    a1 = a.reset_index(drop=True)
    b1 = b.reset_index(drop=True)

    data1, data2 = map(np.asarray, (a1, b1))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)

    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0 * n2)

    cdf_abs_dif = np.absolute(cdf1 - cdf2)
    d = np.max(cdf_abs_dif)

    pos1 = -1
    pos2 = -1
    for i in range(len(cdf_abs_dif)):
        if np.isclose(d, cdf_abs_dif[i]):
            pos1, pos2 = cdf1[i], cdf2[i]
            break

    y_1 = np.arange(n1) / float(n1)
    y_2 = np.arange(n2) / float(n2)

    x_1_idx = []
    for i in range(len(y_1)):
        if np.isclose(pos1, y_1[i]):
            x_1_idx.append(i)
            break
    x_2_idx = []
    for i in range(len(y_2)):
        if np.isclose(pos2, y_2[i]):
            x_2_idx.append(i)
            break

    x_0 = (data1[x_1_idx[0]] + data2[x_2_idx[0]]) / 2.0

    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(data1,y_1,label="good sample")
    plt.plot(data2,y_2,label="bad sample")
    plt.legend(loc='lower right')
    plt.plot([x_0, x_0], [y_1[x_1_idx], y_2[x_2_idx]], linestyle="--")
    plt.scatter([x_0, x_0], [y_1[x_1_idx], y_2[x_2_idx]], 50, color='orange')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Probability')
    plt.ylabel('F_n(Probability)')
    plt.title('Kolmogorov - Smirnov Chart')
    plt.savefig(fig_path)

    return d, x_0, prec(Y_true, Y_predprob, x_0), rec(Y_true, Y_predprob, x_0)


def confusionMatrixs(Y_true, Y_predprob, cut_off):
    vfunc = np.vectorize(greaterThan) 
    y_pred = vfunc(Y_predprob, cut_off)
    return confusion_matrix(Y_true, y_pred)


def keyClassificationMetrics(Y_true, Y_predprob, cut_off):
    vfunc = np.vectorize(greaterThan) 
    y_pred = vfunc(Y_predprob, cut_off)
    target_names = ['class 0', 'class 1']
    return classification_report(Y_true, y_pred, target_names=target_names)


def dividZeroProcess(a, b):
    if b == 0:
        return -1 
    else:
        return a * 1.0000 / b


def GBoddsWRTpredprob(Y_true, Y_predprob, groupCount=10):
    """
    groupCount: 10,20
    should return a table with GBodds per predprob group
    (即把predprob按照quantile分为10或者20组，每组的good/bad ratio)
    """
    test = pd.DataFrame({'label': Y_true, 'predprob': Y_predprob})
    good = []
    bad = []
    oddsRatio = []

    if groupCount == 10:
        quant = []
        for i in range(1, 11, 1):
            quant.append(test['predprob'].quantile(0.1 * i))
        for i in range(1, 11, 1):
            if i == 1:
                pro_1 = quant[i - 1]
                df = test[test.predprob <= pro_1]
                gcount = df[df.label == 0].shape[0]
                bcount = df[df.label == 1].shape[0]           
                good.append(gcount)
                bad.append(bcount)
                oddsRatio.append(dividZeroProcess(gcount,bcount))
            else:
                pro_0 = quant[i - 2]
                pro_1 = quant[i - 1]
                df1 = test[test.predprob > pro_0]
                df = df1[df1.predprob <= pro_1]
                gcount = df[df.label == 0].shape[0]
                bcount = df[df.label == 1].shape[0]           
                good.append(gcount)
                bad.append(bcount)
                oddsRatio.append(dividZeroProcess(gcount,bcount)) 
        oddsDF = pd.DataFrame({'proQuantile': ['10%', '20%', '30%', '40%', '50%',
                                               '60%', '70%', '80%', '90%', '100%'], 
                               'Prob_1': quant,
                               'good_count': good, 'bad_count': bad, 'odds': oddsRatio})
    else:
        quant = []
        for i in range(1, 21, 1):
            quant.append(test['predprob'].quantile(0.05 * i))
        for i in range(1, 21, 1):
            if i==1 :
                pro_1 = quant[i-1]
                df = test[test.predprob <= pro_1]
                gcount = df[df.label == 0].shape[0]
                bcount = df[df.label == 1].shape[0]           
                good.append(gcount)
                bad.append(bcount)
                oddsRatio.append(dividZeroProcess(gcount,bcount))
            else :
                pro_0 = quant[i - 2]
                pro_1 = quant[i - 1]
                df1 = test[test.predprob > pro_0]
                df = df1[df1.predprob <= pro_1]
                gcount = df[df.label == 0].shape[0]
                bcount = df[df.label == 1].shape[0]           
                good.append(gcount)
                bad.append(bcount)
                oddsRatio.append(dividZeroProcess(gcount, bcount))    
        oddsDF = pd.DataFrame({'proQuantile': ['5%', '10%', '15%', '20%', '25%',
                                               '30%', '35%', '40%', '45%', '50%',
                                               '55%', '60%', '65%', '70%', '75%',
                                               '80%', '85%', '90%', '95%', '100%'],
                               'Prob_1': quant,
                               'good_count': good, 'bad_count': bad, 'odds': oddsRatio})
    oddsDF = oddsDF[['proQuantile', 'Prob_1', 'good_count', 'bad_count', 'odds']]
    return oddsDF

