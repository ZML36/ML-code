from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

class ClassificationMetrics:
    def __init__(self, y_true, y_pred, y_prob, n_classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.n_classes = n_classes
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred, average='weighted')

    def precision(self):
        return precision_score(self.y_true, self.y_pred, average='weighted')

    def auc(self):
        one_hot_labels = label_binarize(np.expand_dims(np.array(self.y_true), axis=-1), classes=[i for i in range(self.n_classes)])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i], np.array(self.y_prob)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        return roc_auc