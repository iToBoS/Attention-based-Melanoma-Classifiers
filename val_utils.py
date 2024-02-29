 # -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:01:49 2022

@author: uvl
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix,roc_auc_score,precision_recall_curve,auc
                  
def read_classification(predict,targets,classes_n):
    results = []
    predict = np.array(predict)
    for i in range(len(predict)):
        results.append(np.argmax(predict[i][0:classes_n]))
   
    y_test = targets
    y_pred = results 

    labels = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV']
    if classes_n==9:
            labels = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV','UNK']
    
    report= metrics.classification_report(y_test, y_pred, target_names=labels,output_dict=True)

    pr_ak , rec_ak , _= precision_recall_curve((y_test == labels.index('AK')).astype(float), predict[:, labels.index('AK')])
    pr_bcc, rec_bcc , _ = precision_recall_curve((y_test == labels.index('BCC')).astype(float), predict[:, labels.index('BCC')])
    pr_bkl, rec_bkl , _= precision_recall_curve((y_test == labels.index('BKL')).astype(float), predict[:, labels.index('BKL')])
    pr_df, rec_df , _ = precision_recall_curve((y_test == labels.index('DF')).astype(float), predict[:, labels.index('DF')])
    pr_scc , rec_scc , _ = precision_recall_curve((y_test == labels.index('SCC')).astype(float), predict[:, labels.index('SCC')])
    pr_vasc , rec_vasc , _= precision_recall_curve((y_test == labels.index('VASC')).astype(float), predict[:, labels.index('VASC')])
    pr_mel , rec_mel , _ = precision_recall_curve((y_test == labels.index('MEL')).astype(float), predict[:, labels.index('MEL')])
    pr_nv , rec_nv , _ = precision_recall_curve((y_test == labels.index('NV')).astype(float), predict[:, labels.index('NV')])
    if classes_n==9:
        pr_unk , rec_unk , _ = precision_recall_curve((y_test == labels.index('UNK')).astype(float), predict[:, labels.index('UNK')])
        pr_auc_unk =  auc(rec_unk , pr_unk)
        wandb.log({'val-unk-pr-auc': pr_auc_unk})
    pr_auc_ak =  auc(rec_ak , pr_ak)
    pr_auc_bcc = auc(rec_bcc , pr_bcc)
    pr_auc_bkl= auc(rec_bkl, pr_bkl)
    pr_auc_df = auc(rec_df , pr_df)
    pr_auc_scc = auc(rec_scc , pr_scc)
    pr_auc_vasc = auc(rec_vasc , pr_vasc)
    pr_auc_mel = auc(rec_mel , pr_mel)
    pr_auc_nv = auc(rec_nv , pr_nv)
    #macro
    if classes_n==9:
        pr_auc = (pr_auc_ak + pr_auc_bcc + pr_auc_bkl + pr_auc_df + pr_auc_scc + pr_auc_vasc + pr_auc_mel + pr_auc_nv + pr_auc_unk) / classes_n
    elif classes_n==8:
        pr_auc = (pr_auc_ak + pr_auc_bcc + pr_auc_bkl + pr_auc_df + pr_auc_scc + pr_auc_vasc + pr_auc_mel + pr_auc_nv) / classes_n
    
    
    wandb.log({'val-PR-AUC': pr_auc})
    wandb.log({ 'val_precision':report['weighted avg']['precision'], 
    'val_recall': report['weighted avg']['recall'], 'Mel-F1-val': report['MEL']['f1-score'], 'BCC-F1-val' : report['BCC']['f1-score'], 'AK-F1-val' :report['AK']['f1-score'] , 
    'SCC-F1-val':report['SCC']['f1-score'], 'DF-F1-val':report['DF']['f1-score'],'VASC-F1-val':report['VASC']['f1-score'], 'BKL-F1-val':report['BKL']['f1-score'],
    'NV-F1-val':report['NV']['f1-score']})
    if classes_n==9:
         wandb.log({'UNK-F1-val':report['UNK']['f1-score']})   
    # =============================================================================
    # AUC
    # =============================================================================
    #ROC_AUC = roc_auc_score(y_test,predict[:,:classes_n], multi_class='ovr') this is in the train file
    auc_ak = roc_auc_score((targets == labels.index('AK')).astype(float), predict[:, labels.index('AK')])
    auc_bcc = roc_auc_score((targets == labels.index('BCC')).astype(float), predict[:, labels.index('BCC')])
    auc_bkl = roc_auc_score((targets == labels.index('BKL')).astype(float), predict[:, labels.index('BKL')])
    auc_df = roc_auc_score((targets == labels.index('DF')).astype(float), predict[:, labels.index('DF')])
    auc_scc = roc_auc_score((targets == labels.index('SCC')).astype(float), predict[:, labels.index('SCC')])
    auc_vasc = roc_auc_score((targets == labels.index('VASC')).astype(float), predict[:, labels.index('VASC')])
    auc_mel = roc_auc_score((targets == labels.index('MEL')).astype(float), predict[:, labels.index('MEL')])
    auc_nv = roc_auc_score((targets == labels.index('NV')).astype(float), predict[:, labels.index('NV')])
    if classes_n==9:
        auc_unk = roc_auc_score((targets == labels.index('UNK')).astype(float), predict[:, labels.index('UNK')])
        wandb.log({'val-unk-auc':auc_unk})
   
    wandb.log({'val-ak-auc':auc_ak,'val-bkl-auc':auc_bkl,'val-df-auc':auc_df,'val-vasc-auc':auc_vasc,'val_mel-auc':auc_mel, 'val-bcc-auc':auc_bcc , 'val-scc-auc':auc_scc,'val-nv-auc':auc_nv})
