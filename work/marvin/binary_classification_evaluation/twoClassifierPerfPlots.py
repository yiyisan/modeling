
# coding: utf-8

# In[ ]:

# <api>
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics

from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt 


# In[ ]:

# <api>
# alg: current used model (parent_model)
# alg2: new model iterated on incremental data (child_model)
# testfresh: new model 训练过程中，划分train/test，testfresh: test里面并且是新增的数据
def applyTwoModelsOnDataset(alg_child, alg_parent, testdf, trn_d, parent_end_index, datamapper, cmpsub_fig_path):

    testdf_index = testdf.index.tolist()
    fresh_testdf_index = [index for index in testdf_index if index >= parent_end_index]
    fresh_testdf = trn_d.loc[fresh_testdf_index]

    test_array = datamapper.transform(fresh_testdf)
    test = test_array[:, :-1]
    labels_test = test_array[:, -1]

    prob_child = alg_child.predict_proba(test)[:, 1]
    prob_parent = alg_parent.predict_proba(test)[:, 1]

    mainMetricsComparison((labels_test, prob_child), (labels_test, prob_parent), cmpsub_fig_path)
    return prob_parent, prob_child

# <api>
def mainMetricsComparison(testX, testY, cmpsub_fig_path):
    # plots: KS, ROC, precision_recall, precision_cutoff, recall_cutoff
    rcParams['figure.figsize'] = 10, 10
    plt.subplot(2, 2, 1)
    roc_curve_plotXY(testX, testY)
    plt.subplot(2, 2, 2)
    precision_recall_curve_plotXY(testX, testY)
    plt.subplot(2, 2, 3)
    precision_cutoff_curve(testX, testY)
    plt.subplot(2, 2, 4)
    recall_cutoff_curve(testX, testY)
    plt.savefig(cmpsub_fig_path)

# <api>
def greaterThan(a, b):
    return 1 if a > b else 0


# <api>
def metricsPlotXY(metricX, metricyY, xlab, ylab, title, loc):
    x, y = metricX
    x1, y1 = metricyY
    plt.plot(x, y, label='1st classifier')
    plt.plot(x1, y1, label='2nd classifier')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc=loc)


# <api>
def prec(Y_true, Y_predprob, t): 
    vfunc = np.vectorize(greaterThan) 
    return metrics.precision_score(Y_true, vfunc(Y_predprob, t))


# <api>
def rec(Y_true, Y_predprob, t): 
    vfunc = np.vectorize(greaterThan) 
    return metrics.recall_score(Y_true, vfunc(Y_predprob, t))


# <api>
def auc_calculate(Y_true,Y_predprob):
    fpr,tpr,thresh = roc_curve(Y_true, Y_predprob)
    return round(metrics.auc(fpr, tpr), 4)


# In[ ]:

# <api>
def roc_curve_plotXY(testX, testY):
    Y_trueX, Y_predprobX = testX
    Y_trueY, Y_predprobY = testY
    
    from sklearn.metrics import roc_curve
    fprX, tprX, threshX = roc_curve(Y_trueX, Y_predprobX)
    fprY, tprY, threshY = roc_curve(Y_trueY, Y_predprobY)
    
    aucX = round(auc_calculate(Y_trueX, Y_predprobX), 4)    
    aucY = round(auc_calculate(Y_trueY, Y_predprobY), 4)   
    
    plt.plot(fprX, tprX, label='ROC-AUC overall (1st),\n AUC Score={}'.format(aucX))
    plt.plot(fprY, tprY, label='ROC-AUC overall (2nd),\n AUC Score={}'.format(aucY))
    
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot([0, 1], [0, 1])    
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    #plt.show()


# <api>
def precision_recall_curve_plotXY(testX, testY):
    Y_trueX, Y_predprobX = testX
    Y_trueY, Y_predprobY = testY
    
    precisionX, recallX, thresholdX = precision_recall_curve(Y_trueX, Y_predprobX)
    precisionY, recallY, thresholdY = precision_recall_curve(Y_trueY, Y_predprobY)
    
    metricsPlotXY((precisionX, recallX), (precisionY, recallY),
                  "Precision", "Recall", "Precision Vs Recall Curve", "upper right")


# <api>
def precision_cutoff_curve(testX, testY):
    Y_trueX, Y_predprobX = testX
    Y_trueY, Y_predprobY = testY
    
    vfunc = np.vectorize(greaterThan)
    max_tX = max(Y_predprobX)
    tX = []
    precX = []
    for i in range(0, 101, 1):
        if i / 100.0000 <= max_tX:
            tX.append(i / 100.0000)
            precX.append(metrics.precision_score(Y_trueX, vfunc(Y_predprobX, tX[i])))
            
    max_tY = max(Y_predprobY)
    tY = []
    precY = []
    for i in range(0, 101, 1):
        if i / 100.0000 <= max_tY:
            tY.append(i / 100.0000)
            precY.append(metrics.precision_score(Y_trueY, vfunc(Y_predprobY, tY[i])))
            
    metricsPlotXY((tX, precX), (tY, precY),
                  "cut-off", "Precision", "Precision Vs Cut-off Curve", "lower right")


# <api>  
def recall_cutoff_curve(testX, testY):
    (Y_trueX, Y_predprobX) = testX
    (Y_trueY, Y_predprobY) = testY

    vfunc = np.vectorize(greaterThan) 
    vfunc = np.vectorize(greaterThan) 
    max_tX = max(Y_predprobX)
    tX = []
    recallX = []
    for i in range(0, 101, 1):
        if i / 100.0000 <= max_tX:
            tX.append(i / 100.0000)
            recallX.append(metrics.recall_score(Y_trueX, vfunc(Y_predprobX, tX[i])))

    max_tY = max(Y_predprobY)
    tY = []
    recallY = []
    for i in range(0, 101, 1):
        if i / 100.0000 <= max_tY:
            tY.append(i / 100.0000)
            recallY.append(metrics.recall_score(Y_trueY, vfunc(Y_predprobY, tY[i])))

    metricsPlotXY((tX, recallX), (tY, recallY),
                  "cut-off", "Recall", "Recall Vs Cut-off Curve", "upper right")

