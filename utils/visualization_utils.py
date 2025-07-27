# %%
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, confusion_matrix, \
     RocCurveDisplay, precision_score, recall_score, average_precision_score, PrecisionRecallDisplay, precision_recall_curve
from sklearn.preprocessing import label_binarize
from scipy import interp
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from tqdm import tqdm
import json
import torch
import os
from icecream import ic
ic.disable()

from torchvision import transforms

import monai
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    Resized,
)
# %%

def visualize_heads(writer, att_map, cols, step, num):
    to_shows = []
    batch_num = att_map.shape[0]
    head_num = att_map.shape[1]
    # att_map = att_map.squeeze()
    for i in range(batch_num):
        for j in range(head_num):
            to_shows.append((att_map[i][j], f'Batch {i} Head {j}'))
        average_att_map = att_map[i].mean(axis=0)
        to_shows.append((average_att_map, f'Batch {i} Head Average'))

    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])

    writer.add_figure("attention_{}".format(num), fig, step)
    # plt.show()

def read_csv(filename):
    return pd.read_csv(filename)

def write_scores(f, preds, labels, masks=None, fnames=None):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    if masks is not None:
        masks = masks.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        line = pred + '__' + label
        if masks is not None:
            line += '__' + ''.join([str(m) for m in masks[index]])
            ic(line)
        if fnames:
            fname = fnames[index]
            f.write(line + '__' + fname + '\n')
        else:
            f.write(line + '\n')

def read_raw_score(txt_file, mask=False):
    labels, scores = [], []
    if mask:
        masks = []
    with open(txt_file, 'r') as f:
        for line in f:
            vals = line.strip('\n').split('__')
            if mask:
                masks.append([int(v) for v in vals[-1]])
                label = vals[-2]
                probs = vals[:-2]

            else:
                label = vals[-1]
                probs = vals[:-1]
            ic(type(probs))
            scores.append([float(p) for p in probs])
            labels.append(int(label))
    if mask:
        return np.array(labels), np.array(scores), np.array(masks)
    return np.array(labels), np.array(scores)

def get_classification_report(y_true, y_pred, features, output_dict=True):
    report = classification_report(
        y_true,
        y_pred,
        output_dict=output_dict,
        target_names=features
    )
    return report

def plot_classification_report(report, filepath, format):
    figure(figsize=(10, 8), dpi=100)
    cls_report_plot = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
    # plt.show()
    cls_report_plot.figure.savefig(filepath, format=format, dpi=300, bbox_inches='tight')

# Confusion Matrix

def confusion_matrix_2_2(y_true, y_pred, labels=[0,1]):
    return confusion_matrix(y_true, y_pred, labels=labels)

def plot_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=10):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=True, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for class - " + class_label)

# def confusion_matrix_3_3(y_true, y_pred):
#     return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

def multilabel_cm(y_true, y_pred):
    return multilabel_confusion_matrix(y_true, y_pred)

def plot_multilabel_cm(y_true, y_pred, features, filepath, format, class_names=[0,1]):
    cm = multilabel_cm(y_true, y_pred)
    fig, ax = plt.subplots(4, 4, figsize=(10, 8), dpi=100)

    for axes, cfs_matrix, label in zip(ax.flatten(), cm, features):
        plot_confusion_matrix(cfs_matrix, axes, label, class_names)

    fig.tight_layout()
    # plt.show()
    fig.figure.savefig(filepath, format=format, dpi=300, bbox_inches='tight')

# AUC ROC

def plot_curve_multiple(combinations, Xs, Ys, avg_scores, numbers, metric, title, out_file, modalities, lw=1, text_size=14):
    with plt.style.context('seaborn-deep'):    
        fig, ax = plt.subplots(dpi=100)
        colors = ['g', 'b', 'r', 'm','c']
        linestyles = ['-.', ':', '-.', '--', '-.', '-']
        hatches = ['//////', '....', '||||||', '*****', '//////', '.....']
        sorted_indices = np.argsort(avg_scores)[::-1]
        ic(len(avg_scores), sorted_indices)
        # exit()
        for idx in sorted_indices:
            comb = combinations[idx]
            ic(comb)
            label = "+".join([modalities[i] for i in range(len(comb)) if comb[i] == 1]) + ' (n={0}) ({1} = {2:0.2f})'.format(numbers[idx], metric, avg_scores[idx])
            ic(label)
            ax.plot(Xs[idx], Ys[idx], lw=lw, alpha=1, label=label)
            ax.plot([0, 1], [0, 1], 'k--', lw=lw/2, alpha=0.2)

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        legend_properties = {'weight': 'normal', 'size': text_size}
        ax.legend(loc="lower right", prop=legend_properties)
        if metric == 'AUC':
            xlabel = '1-Specificity (FPR)'
            ylabel = 'Sensitivity (TPR)'
        else:
            xlabel = 'Recall'
            ylabel = 'Precision'
        ax.set_xlabel(xlabel, fontsize=text_size, fontweight='normal')
        ax.set_ylabel(ylabel, fontsize=text_size, fontweight='normal')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(title, fontsize=20, fontweight='bold')

    fig.savefig(out_file, bbox_inches='tight')
    ic(out_file)
    fig.clf()
    plt.close()
        

def roc_auc(y_true, y_pred):
    ic(y_pred.shape)
    n_classes = y_pred.shape[1]

    tpr = dict()
    fpr = dict()
    auc_scores = dict()
    thresholds = dict()
    y_true = label_binarize(y_true, classes=[i for i in range(n_classes)])
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true=y_true[:, i], y_score=y_pred[:, i], pos_label=1, drop_intermediate=False)
        auc_scores[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true.ravel(), y_pred.ravel()
    )
    auc_scores["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    # fpr["macro"] = np.mean(list(fpr.values())[:n_classes], axis=0)
    # tpr["macro"] = np.mean(list(tpr.values())[:n_classes], axis=0)
    auc_scores["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute weighted-average ROC curve and ROC area
    support = np.sum(y_true, axis=0)
    weights = support / np.sum(support)
    weighted_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        weighted_tpr += weights[i] * np.interp(all_fpr, fpr[i], tpr[i])
    fpr["weighted"] = all_fpr
    tpr["weighted"] = weighted_tpr
    auc_scores["weighted"] = auc(fpr["weighted"] , tpr["weighted"])  

    return fpr, tpr, auc_scores, thresholds


def generate_roc_binary(y_true, y_pred, labels_list, save_path, suffix='', dset='NACC'):
    y_pred = y_pred[:,1]
    fpr, tpr, thresh = roc_curve(y_true=y_true, y_score=y_pred, drop_intermediate=True)
    auc_score = auc(fpr, tpr)
    fpr_value = np.linspace(0, 1, 100)
    interp_tpr = np.interp(fpr_value, fpr, tpr)
    plt.figure()
    lw = 2
    text_size = 10
    colors = ['darkorange', 'steelblue', 'green', "yellow", "red", "black", "brown", "pink", "cyan", "purple", "lime", "aqua", "gold", "skyblue"] # set the colors for each class
    color = colors[0]
    plt.plot(fpr_value, interp_tpr, color=color, lw=lw/2, alpha=0.8,
            label='(AUC = {0:0.2f})'.format(auc_score))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    legend_properties = {'size': text_size}
    plt.legend(loc="lower right", prop=legend_properties)
    # plt.show()
    plt.savefig(os.path.join(save_path,f"{dset}_ROC_curve{suffix}.png"), format='png', dpi=300, bbox_inches='tight')
    ic(os.path.join(save_path,f"{dset}_ROC_curve{suffix}.png"))
    return auc_score

def generate_pr_binary(y_true, y_pred, labels_list, save_path, suffix='', dset='NACC'):
    y_pred = y_pred[:,1]
    pr, rc, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    average_precision = average_precision_score(y_true=y_true, y_score=y_pred)
    precision, recall = pr[::-1], rc[::-1]
    
    plt.figure()
    lw = 2
    text_size = 10
    colors = ['darkorange', 'steelblue', 'green', "yellow", "red", "black", "brown", "pink", "cyan", "purple", "lime", "aqua", "gold", "skyblue"] # set the colors for each class
    color = colors[0]
    mean_recall = np.linspace(0, 1, 100)
    interp_precision = np.interp(mean_recall, recall, precision)
    plt.plot(mean_recall, interp_precision, color=color, lw=lw/2, alpha=0.8,
    label='(AP = {0:0.2f})'.format(average_precision))

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_path,f"{dset}_PR_curve{suffix}.png"), format='png', dpi=300, bbox_inches='tight')
    ic(os.path.join(save_path,f"{dset}_PR_curve{suffix}.png"))
    return average_precision
    
def generate_roc(y_true, y_pred, features, save_path, suffix='', dset='NACC'):
    fpr, tpr, auc_scores, _ = roc_auc(y_true=y_true, y_pred=y_pred)
    n_classes = y_pred.shape[1]

    # Individual ROC curves
    plt.figure()
    lw = 2
    text_size = 10
    colors = ['darkorange', 'steelblue', 'green', "yellow", "red", "black", "brown", "pink", "cyan", "purple", "lime", "aqua", "gold", "skyblue"] # set the colors for each class
    colors = colors[:len(features)]
    fpr_value = np.linspace(0, 1, 100)
    for i, color in zip(range(n_classes), colors):
        interp_tpr = np.interp(fpr_value, fpr[i], tpr[i])
        plt.plot(fpr_value, interp_tpr, color=color, lw=lw/2, alpha=0.8,
                label='{0} (AUC = {1:0.2f})'.format(features[i], auc_scores[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    legend_properties = {'size': text_size}
    plt.legend(loc="lower right", prop=legend_properties)
    # plt.show()
    plt.savefig(os.path.join(save_path,f"{dset}_ROC_curves{suffix}.png"), format='png', dpi=300, bbox_inches='tight')

    # Average ROC curves
    plt.figure(dpi=100)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (AUC = {0:0.2f})'
            ''.format(auc_scores["micro"]),
            color='deeppink', linestyle='--', linewidth=lw/2, alpha=0.8)

    plt.plot(fpr["macro"], tpr["macro"],
        label='macro-average ROC curve (AUC = {0:0.2f})'
        ''.format(auc_scores["macro"]),
        color='navy', linestyle='--', linewidth=lw/2, alpha=0.8)

    plt.plot(fpr["weighted"], tpr["weighted"],
        label='weighted-average ROC curve (AUC = {0:0.2f})'
        ''.format(auc_scores["weighted"]),
        color='darkorange', linestyle='--', linewidth=lw/2, alpha=0.8)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC curves')
    plt.legend(loc="lower right", prop=legend_properties)
    # plt.show()
    plt.savefig(os.path.join(save_path,f"{dset}_Average_ROC_curves{suffix}.png"), format='png', dpi=300, bbox_inches='tight')
    return auc_scores

# P-R curve

def precision_recall(y_true, y_pred):
    # Compute the precision-recall curve and average precision for each class
    n_classes = y_pred.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        precision[i], recall[i] = precision[i][::-1], recall[i][::-1]
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    # Compute the micro-average precision-recall curve and average precision
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                        average="micro")

    # Compute the macro-average precision-recall curve and average precision
    mean_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
    # mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    for i in range(n_classes):
        mean_precision += np.interp(mean_recall, recall[i], precision[i])
    mean_precision /= n_classes
    recall["macro"] = mean_recall
    precision["macro"] = mean_precision

    average_precision["macro"] = average_precision_score(y_true, y_pred,
                                                        average="macro")

    # Compute the weighted-average precision-recall curve and average precision

    support = np.sum(y_true, axis=0)
    weights = support / np.sum(support)
    weighted_precision = np.zeros_like(mean_recall)
    for i in range(n_classes):
        weighted_precision += weights[i] * np.interp(mean_recall, recall[i], precision[i])
    recall["weighted"] = mean_recall
    precision["weighted"] = weighted_precision
    average_precision["weighted"] = average_precision_score(y_true, y_pred,
                                                                average="weighted")

    return precision, recall, average_precision


def generate_pr(y_true, y_pred, features, save_path, suffix='', dset='NACC'):
    n_classes = y_pred.shape[1]
    y_true = label_binarize(y_true, classes=[i for i in range(n_classes)])
    precision, recall, average_precision = precision_recall(y_true=y_true, y_pred=y_pred)
    lw = 2
    text_size = 10

    # Plot the precision-recall curves for all classes, micro-average, macro-average, and weighted-average
    plt.figure()

    colors = ['darkorange', 'steelblue', 'green', "yellow", "red", "black", "brown", "pink", "cyan", "purple", "lime", "aqua", "gold", "skyblue"] # set the colors for each class
    mean_recall = np.linspace(0, 1, 100)
    for i, color in zip(range(n_classes), colors):
        interp_precision = np.interp(mean_recall, recall[i], precision[i])
        plt.plot(mean_recall, interp_precision, color=color, lw=lw/2, alpha=0.8,
        label='{0} (AP = {1:0.2f})'.format(features[i], average_precision[i]))

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_path, f"{dset}_PR_curves{suffix}.png"), format='png', dpi=300, bbox_inches='tight')
    # plt.show()

    plt.figure()

    plt.plot(recall['micro'], precision['micro'],
            label='micro-average Precision-Recall curve (AP = {0:0.2f})'.format(average_precision["micro"])
            , linestyle='--', linewidth=lw/2, alpha=0.8)
    plt.plot(recall['macro'], precision['macro'], 
            label='macro-average Precision-Recall curve (AP = {0:0.2f})'.format(average_precision["macro"])
            , linestyle='--', linewidth=lw/2, alpha=0.8)
    plt.plot(recall['weighted'], precision['weighted'], 
            label='weighted-average Precision-Recall curve (AP = {0:0.2f})'.format(average_precision["weighted"])
            , linestyle='--', linewidth=lw/2, alpha=0.8)


    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision-Recall curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_path,f"{dset}_Average_PR_curves{suffix}.png"), format='png', dpi=300, bbox_inches='tight')
    # plt.show()
    return average_precision

def save_performance_report(met, filepath):
    figure(figsize=(24, 20), dpi=300)
    met_df = pd.DataFrame(met).transpose()
    report_plot = sns.heatmap(met_df, annot=True)

    # plt.show()
    report_plot.figure.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')

#%%
def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

flip_and_jitter = monai.transforms.Compose([
        monai.transforms.RandAxisFlipd(keys=["image"], prob=0.5),
        transforms.RandomApply(
            [
                monai.transforms.RandAdjustContrastd(keys=["image"], gamma=(-0.3,0.3)), # Random Gamma => randomly change contrast by raising the values to the power log_gamma 
                monai.transforms.RandBiasFieldd(keys=["image"]), # Random Bias Field artifact
                monai.transforms.RandGaussianNoised(keys=["image"]),

            ],
            p=0.4
        ),
    ])

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        # self.problematic_indices = []
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.7, max_roi_scale=1, random_size=True, random_center=True),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=128),
                flip_and_jitter,
                monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
                minmax_normalized,
            ]            
        )
        
        self.vld_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                # CenterSpatialCropd(keys=["image"], roi_size=(args.img_size,)*3),
                Resized(keys=["image"], spatial_size=(128*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=128),
                minmax_normalized,
            ]
        )
        
        if dat_type == 'trn':
            self.transforms = self.train_transforms
        else:
            self.transforms = self.vld_transforms

    def __call__(self, data):
        image_data = data["image"]
        try:
            # check = nib.load(image_data).get_fdata()
            # print(len(check.shape))
            # if len(check.shape) > 3:
            #     return None
            
            return self.transforms(data)
        except Exception as e:
            print(f"Error processing image: {image_data}{e}")
            return None
        
# tst_filter_transform = FilterImages(dat_type='tst')
tst_filter_transform = None


# #%%
# def save_predictions(dat_tst, y_true, scores_proba, scores, save_path=None, filename=None, if_save=True):
#     y_true = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_tst.labels]
#     mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]

#     y_true_ = {f'{k}_label': [smp[k] for smp in y_true] for k in y_true[0] if k in dat_file.columns}
#     scores_proba_ = {f'{k}_prob': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores_proba)] for k in scores_proba[0] if k in dat_file.columns}
#     scores_ = {f'{k}_logit': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores)] for k in scores[0] if k in dat_file.columns}
#     cdr = dat_file['cdr_CDRGLOB']
#     ids = dat_file['ID']

#     y_true_df = pd.DataFrame(y_true_)
#     scores_df = pd.DataFrame(scores_)
#     scores_proba_df = pd.DataFrame(scores_proba_)
#     cdr_df = pd.DataFrame(cdr)
#     id_df = pd.DataFrame(ids)
#     if 'fhs' in fname:
#         fhsid = ids = dat_file[['id', 'idtype', 'framid']]
#         fhsid_df = pd.DataFrame(fhsid)
#         df = pd.concat([fhsid_df, id_df, y_true_df, scores_proba_df, cdr_df], axis=1)
#     else:
#         df = pd.concat([id_df, y_true_df, scores_proba_df, cdr_df], axis=1)
#     if if_save:
#         df.to_csv(save_path + filename, index=False)
#     return df
