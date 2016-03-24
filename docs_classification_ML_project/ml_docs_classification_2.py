# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 22:52:40 2016

@author: RaoUmer
"""
print (__doc__)

# importing required modules
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

# classifier modules
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn import svm#, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
#from pprint import pprint
from time import time


def prepare_data(data_train, data_test):
    # Converting text to vectors of numerical values using tf-idf to form feature vector
    vectorizer = TfidfVectorizer()
    data_train_vectors = vectorizer.fit_transform(data_train.data)
    data_test_vectors = vectorizer.transform(data_test.data)    
    
    return data_train_vectors, data_test_vectors

def MNB(data_train, data_train_vectors, data_test_vectors, **kwargs):
    # Implementing classification model- using MultinomialNB
    clf_MNB = MultinomialNB(alpha=.01)
    clf_MNB.fit(data_train_vectors, data_train.target)
    y_pred = clf_MNB.predict(data_test_vectors)
    
    return y_pred

def BNB(data_train, data_train_vectors, data_test_vectors, **kwargs):
    # Implementing classification model- using BernoulliNB
    clf_BNB = BernoulliNB(alpha=.01)
    clf_BNB.fit(data_train_vectors, data_train.target)
    y_pred = clf_BNB.predict(data_test_vectors)
    
    return y_pred

def KNN(data_train, data_train_vectors, data_test_vectors, **kwargs):
    # Implementing classification model- using KNeighborsClassifier
    clf_knn =  KNeighborsClassifier(n_neighbors=5)
    clf_knn.fit(data_train_vectors, data_train.target)
    y_pred = clf_knn.predict(data_test_vectors)
    
    return y_pred

def NC(data_train, data_train_vectors, data_test_vectors, **kwargs):
    # Implementing classification model- using NearestCentroid
    clf_nc =  NearestCentroid()
    clf_nc.fit(data_train_vectors, data_train.target)
    y_pred = clf_nc.predict(data_test_vectors)
    
    return y_pred

def SVM(data_train, data_train_vectors, data_test_vectors, **kwargs):
    # Implementing classification model- using LinearSVC
    clf_svc =  LinearSVC()
    clf_svc.fit(data_train_vectors, data_train.target)
    y_pred_score = clf_svc.decision_function(data_test_vectors)
    
    return y_pred_score

def PERCEPTRON(data_train, data_train_vectors, data_test_vectors, **kwargs):
    # Implementing classification model- using Perceptron
    clf_p =  Perceptron()
    clf_p.fit(data_train_vectors, data_train.target)
    y_pred = clf_p.predict(data_test_vectors)
    
    return y_pred

def RF(data_train, data_train_vectors, data_test_vectors, **kwargs):
    # Implementing classification model- using RandomForestClassifier
    clf_rf =  RandomForestClassifier()
    clf_rf.fit(data_train_vectors, data_train.target)
    y_pred = clf_rf.predict(data_test_vectors)
    
    return y_pred

def SGD(data_train, data_train_vectors, data_test_vectors, **kwargs):
    # Implementing classification model- using SGDClassifier
    clf_sgd =  SGDClassifier()
    clf_sgd.fit(data_train_vectors, data_train.target)
    y_pred = clf_sgd.predict(data_test_vectors)
    
    return y_pred

def evaluation_score(data_test, y_pred, **kwargs):
    avg = kwargs.pop('average','binary')
    print "F1-measure:",metrics.f1_score(data_test.target, y_pred, average=avg)
    print "Testing Accuracy:",metrics.accuracy_score(data_test.target,y_pred)
    print "Confusion Matrix:\n",metrics.confusion_matrix(data_test.target, y_pred)
    print "Sensitivity:",metrics.recall_score(data_test.target, y_pred) 
    print "Precision:",metrics.precision_score(data_test.target, y_pred)
    return 0

def ROC_binary_class(data_test, y_pred_score):
    
    fpr, tpr, thresholds = metrics.roc_curve(data_test.target, y_pred_score)
#    print "fpr:",fpr
#    print "tpr:",tpr
    print"AUC-ROC Score:", metrics.roc_auc_score(data_test.target, y_pred_score)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)    
    return 0

def PR_binary_class(data_test, y_pred_score):
    
    precision, recall, _ = precision_recall_curve(data_test.target, y_pred_score)
    #print"AUC-ROC Score:", metrics.roc_auc_score(data_test.target, y_pred)
    plt.plot(precision, recall)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('PR curve for classifier')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)    
    return 0

def ROC_multi_class(data_train, data_test, data_test_vectors):
    
    # Binarize the output
    y_train_label = label_binarize(data_train.target, classes=[0, 1, 2])
    n_classes = y_train_label.shape[1]
    
    
    random_state = np.random.RandomState(1)
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_train_vectors, y_train_label, test_size=.5,
                                                        random_state=0)
    
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    classifier.fit(X_train, y_train)
    y_pred_score = classifier.decision_function(data_test_vectors)
    
    y_test_label = label_binarize(data_test.target, classes=[0, 1, 2])
    
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], y_pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), y_pred_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    # Plot ROC curves for the multiclass problem
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
#    plt.plot(fpr["micro"], tpr["micro"],
#             label='micro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["micro"]),
#             linewidth=2)
#    
#    plt.plot(fpr["macro"], tpr["macro"],
#             label='macro-average ROC curve (area = {0:0.2f})'
#                   ''.format(roc_auc["macro"]),
#             linewidth=2)
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    return 0

def PR_multi_class(data_train, data_test, data_test_vectors):
    # Binarize the output
    y_train_label = label_binarize(data_train.target, classes=[0, 1, 2])
    n_classes = y_train_label.shape[1]
    
    random_state = np.random.RandomState(0)
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_train_vectors, y_train_label, test_size=.5,
                                                        random_state=random_state)
    
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    classifier.fit(X_train, y_train)
    y_pred_score = classifier.decision_function(data_test_vectors)
    
    y_test_label = label_binarize(data_test.target, classes=[0, 1, 2])
    
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_label[:, i], y_pred_score[:, i])
        average_precision[i] = average_precision_score(y_test_label[:, i], y_pred_score[:, i])
    
    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_label.ravel(), y_pred_score.ravel())
    average_precision["micro"] = average_precision_score(y_test_label, y_pred_score, average="micro")
    
    # Plot Precision-Recall curve for each class
    plt.clf()
#    plt.plot(recall["micro"], precision["micro"],
#             label='micro-average PR curve (area = {0:0.2f})'
#                   ''.format(average_precision["micro"]))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],
                 label='PR curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve of multi-class')
    plt.legend(loc="lower right")
    plt.show()
    return 0

def best_parameters_selection():
    return 0


def benchmark(clf, data_train, data_test):
    
    y_train = data_train.target
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(data_train.data)     
    X_test = vectorizer.transform(data_test.data)
    
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    t0 = time()
    clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    clf_descr = str(clf).split('(')[0]
    return clf_descr, train_time, test_time

def plot_benchmark(clf, data_train, data_test):
    #results = []
    #results.append(benchmark(clf, data_train, data_test))
    
    #indices = np.arange(len(results))
    
    #results = [[x[i] for x in results] for i in range(1)]
    
    clf_names, training_time, test_time = benchmark(clf, data_train, data_test)
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)
    
    plt.figure(figsize=(12, 8))
    #plt.title("Score")
    #plt.barh(indices, score, .2, label="score", color='r')
    plt.barh( .3, training_time, .2, label="training time", color='g')
    plt.barh( .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    
    plt.text(-.3,0 ,clf_names)
    
    plt.show()
    return 0
    

if __name__=='__main__':
    
    # Load some categories from the data set
    categories = ['alt.atheism', 'comp.graphics']#, 'sci.space']
    # Uncomment the following to do the analysis on all the categories
    #categories = None
    
    print("Loading 20 newsgroups dataset for categories:")
    print(categories)
    
    # Training data
    data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
    print("%d Training documents" % len(data_train.filenames))
    print("%d Training categories" % len(data_train.target_names))
    print "\n"
    
    # Testing data
    data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
    print("%d Testing documents" % len(data_test.filenames))
    print("%d Testing categories" % len(data_test.target_names))
    print "\n"
    
    data_train_vectors, data_test_vectors = prepare_data(data_train,data_test)
    print data_train_vectors.shape,data_test_vectors.shape
    
    y_pred_score = SVM(data_train, data_train_vectors,data_test_vectors)
    #print y_pred.shape
    
    #evaluation_score(data_test, y_pred_score)#, average='weighted')
    plt.figure(121)
    ROC_binary_class(data_test, y_pred_score)
    
    plt.figure(122)    
    PR_binary_class(data_test, y_pred_score)
    
    plt.figure(123)
    ROC_multi_class(data_train, data_test, data_test_vectors)
    
    plt.figure(124)    
    PR_multi_class(data_train, data_test, data_test_vectors)
    
    #plot_benchmark(LinearSVC(), data_train, data_test)
    #benchmark(LinearSVC(), data_train, data_test)