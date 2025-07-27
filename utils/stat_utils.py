import os
import sys
import numpy as np
from .visualization_utils import generate_roc, generate_pr_binary, generate_roc_binary, generate_pr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
import collections

def get_metrics(y_true, y_prob, y_pred, labels_list, save_path=None, test=False, suffix='', dset='NACC'):
    num_classes = max(2,len(labels_list))
    met = {}
    ic(y_pred)
    ic(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    assert cm.shape == (num_classes, num_classes)

    if num_classes <= 2:
        TN, FP, FN, TP = np.ravel(cm)
    else:
        FP = cm.sum(axis=0) - np.diag(cm) 
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    met['TPR'] = TPR = TP/(TP+FN + sys.float_info.epsilon)
    # Specificity or true negative rate
    met['TNR'] = TNR = TN/(TN+FP + sys.float_info.epsilon) 
    # Precision or positive predictive value
    met['PREC'] = PREC = TP/(TP+FP + sys.float_info.epsilon)
    # Negative predictive value
    met['NPV'] = TN/(TN+FN + sys.float_info.epsilon)
    # Fall out or false positive rate
    met['FPR'] = FP/(FP+TN + sys.float_info.epsilon)
    # False negative rate
    met['FNR'] = FN/(TP+FN + sys.float_info.epsilon)
    # False discovery rate
    met['FDR'] = FP/(TP+FP + sys.float_info.epsilon)
    # Threat score (TS) or Critical Success Index (CSI)
    met['TS'] = TP / (TP + FP + FN + sys.float_info.epsilon)
    # Overall accuracy for each class
    met['ACC'] = ACC = (TP+TN)/(TP+FP+FN+TN + sys.float_info.epsilon)
    met['TOTAL_ACC'] = TOTAL_ACC = np.diag(cm).sum() / cm.sum()
    if num_classes > 2:
        met['CLS_BALANCED_ACC'] = CLS_BALANCED_ACC = TPR.sum() / num_classes
        # Calculate the classification error rate for each class
        class_error_rates = []
        for i in range(num_classes):
            # Total instances for the class (row sum)
            total_instances = np.sum(cm[i, :])
            
            # Misclassified instances for the class (sum of all elements in the row except the diagonal element)
            misclassified = total_instances - cm[i, i]
            
            # Error rate for the class
            error_rate = misclassified / total_instances if total_instances > 0 else 0
            class_error_rates.append(error_rate)

        # Compute the Average Classification Error Rate (ACER)
        met['ACER'] = np.mean(class_error_rates)
    else:
        met['CLS_BALANCED_ACC'] = CLS_BALANCED_ACC = (TPR + TNR) / 2
        met['ACER'] = (met['FNR'] + met['FPR']) / 2

    met['F1'] = F1 = 2 * PREC * TPR / (PREC + TPR + sys.float_info.epsilon)
    met['MCC'] = MCC = (TP*TN - FP*FN) / (np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) + 0.000000001)
    met['sk_MCC'] = matthews_corrcoef(np.asarray(y_true), np.asarray(y_pred))
    if num_classes > 2:
        met['MACRO_F1'] = MACRO_F1 = F1.sum() / num_classes
        met['WEIGHTED_F1'] = WEIGHTED_F1 = (F1 * cm.sum(axis=0)).sum() / cm.sum() 
        met['MACRO_MCC'] = MACRO_MCC = MCC.sum() / num_classes
        met['WEIGHTED_MCC'] = WEIGHTED_MCC = (MCC * cm.sum(axis=0)).sum() / cm.sum()
    

    if test:
        print(cm)
        if len(labels_list) == 1:
            labels_list = [f'Not {labels_list[0]}', labels_list[0]]
        if save_path:
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_list).plot()
            display.figure_.savefig(os.path.join(save_path,f'{dset}_conf_mat{suffix}.png'), dpi=300)
            ic(os.path.join(save_path,f'{dset}_conf_mat{suffix}.png'))
        if num_classes <= 2:
            auc_score = generate_roc_binary(y_true, y_prob, labels_list, save_path, suffix=suffix, dset=dset)
            average_precision = generate_pr_binary(y_true, y_prob, labels_list, save_path, suffix=suffix, dset=dset)
        else:
            auc_score = generate_roc(y_true, y_prob, labels_list, save_path, suffix=suffix, dset=dset)
            average_precision = generate_pr(y_true, y_prob, labels_list, save_path, suffix=suffix, dset=dset)

        met['AUC'] = auc_score
        met['AP'] = average_precision


    # print_metrics_multitask(met)
    for k in met:
        # if k not in ['MACRO_F1','WEIGHTED_F1','MACRO_MCC','WEIGHTED_MCC','TOTAL_ACC']:
        if isinstance(met[k], collections.abc.Sequence) or isinstance(met[k], np.ndarray):
            msg = '{}:\t' + '{:.4f}    ' * len(met[k])
            val = list(met[k])
            msg = msg.format(k, *val)
            msg = msg.replace('nan', '------')
            print(msg.expandtabs(20))
        else:
            print(f"{k}: {met[k]}")

    return met

def get_metrics_multitask(y_true, scores, y_pred, labels_list, save_path=None, test=False, suffix=''):
    ''' ... '''
    met = []
    for i in range(len(y_true[0])):
        met.append(get_metrics(
            y_true[:, i],
            scores[:, i],
            y_pred[:, i],
            labels_list=labels_list, 
            save_path=save_path,
            test=test,
            suffix=suffix,

        ))
    return met

def print_metrics_multitask(met):
    ''' ... '''
    for k in met[0]:
        if k not in ['Confusion Matrix']:
            msg = '{}:\t' + '{:.4f}    ' * len(met)
            val = [met[i][k] for i in range(len(met))]
            msg = msg.format(k, *val)
            msg = msg.replace('nan', '------')
            print(msg.expandtabs(20))