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
from sklearn.metrics import confusion_matrix,roc_auc_score,balanced_accuracy_score,precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight

def class_weights(class_counts):
    total_count = np.sum(class_counts)
    class_props = class_counts / total_count
    class_weights = 1.0 / class_props
    class_weights /= np.sum(class_weights)
    return class_weights
                      
def read_classification(probs,targets,classes_n=8):
    if targets: 
        probs = get_tta_probs(probs,classes_n)    
        results = []
        names =[]
        y_test = []
        names = probs[:,classes_n]

        for i  in range(len(names)):
             y_test.append(int(names[i][0]))

        probs[:,classes_n] = y_test
     
        if classes_n==8:
            mask = probs[:, classes_n] == 8
            predict = probs[~mask]

        y_test = list(predict[:,classes_n])
        print(predict.shape)
        for i in range(len(predict)):
            pred = np.argmax(predict[i][0:classes_n])
            results.append(pred)

        y_pred = results 
        #print(y_test) 
        #print(y_pred)
        print(set(y_test))
        print(set(y_pred))
        y_test= np.array(y_test)
        y_pred= np.array(y_pred)
        labels = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV']

        if classes_n==9:
                labels = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV','UNK']
        
        report= metrics.classification_report(y_test, y_pred, target_names=labels,output_dict=True)
        test_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
      
        pr_ak , rec_ak , _= precision_recall_curve((y_test == labels.index('AK')).astype(float), predict[:, labels.index('AK')])
        pr_bcc, rec_bcc , _ = precision_recall_curve((y_test == labels.index('BCC')).astype(float), predict[:, labels.index('BCC')])
        pr_bkl, rec_bkl , _= precision_recall_curve((y_test == labels.index('BKL')).astype(float), predict[:, labels.index('BKL')])
        pr_df, rec_df , _ = precision_recall_curve((y_test == labels.index('DF')).astype(float), predict[:, labels.index('DF')])
        pr_scc , rec_scc , _ = precision_recall_curve((y_test == labels.index('SCC')).astype(float), predict[:, labels.index('SCC')])
        pr_vasc , rec_vasc , _= precision_recall_curve((y_test == labels.index('VASC')).astype(float), predict[:, labels.index('VASC')])
        pr_mel , rec_mel , _ = precision_recall_curve((y_test == labels.index('MEL')).astype(float), predict[:, labels.index('MEL')])
        pr_nv , rec_nv , _ = precision_recall_curve((y_test == labels.index('NV')).astype(float), predict[:, labels.index('NV')])

        pr_auc_ak =  auc(rec_ak , pr_ak)
        pr_auc_bcc = auc(rec_bcc , pr_bcc)
        pr_auc_bkl= auc(rec_bkl, pr_bkl)
        pr_auc_df = auc(rec_df , pr_df)
        pr_auc_scc = auc(rec_scc , pr_scc)
        pr_auc_vasc = auc(rec_vasc , pr_vasc)
        pr_auc_mel = auc(rec_mel , pr_mel)
        pr_auc_nv = auc(rec_nv , pr_nv)
 
        unique_samples, sample_counts = np.unique(y_test, return_counts=True)
        weights = class_weights(sample_counts)
        w_pr_auc = (weights[labels.index('AK')]* pr_auc_ak + weights[labels.index('BCC')]*pr_auc_bcc +
                   weights[labels.index('BKL')]*pr_auc_bkl + weights[labels.index('DF')]*pr_auc_df +
                   weights[labels.index('SCC')]* pr_auc_scc + weights[labels.index('VASC')]*pr_auc_vasc +
                   weights[labels.index('MEL')]*pr_auc_mel + weights[labels.index('NV')]* pr_auc_nv) / classes_n
        
        pr_auc = (pr_auc_ak + pr_auc_bcc + pr_auc_bkl + pr_auc_df + pr_auc_scc + pr_auc_vasc + pr_auc_mel + pr_auc_nv) / classes_n
        
        wandb.log ({'W-PR-AUC': w_pr_auc,'PR-AUC': pr_auc})
        wandb.log({'AK-PR-AUC': pr_auc_ak, 'BCC-PR-AUC': pr_auc_bcc,'BKL-PR-AUC': pr_auc_bkl,'DF-PR-AUC': pr_auc_df,
                   'SCC-PR-AUC': pr_auc_scc,'MEL-PR-AUC': pr_auc_mel,'NV-PR-AUC': pr_auc_nv,'VASC-PR-AUC': pr_auc_vasc})
      
        wandb.log({'test-f1': test_f1,'test_precision':report['weighted avg']['precision'], 
        'test_recall': report['weighted avg']['recall'], 'Mel-F1': report['MEL']['f1-score'], 'BCC-F1' : report['BCC']['f1-score'], 'AK-F1' :report['AK']['f1-score'] , 
        'SCC-F1':report['SCC']['f1-score'], 'DF-F1':report['DF']['f1-score'],'VASC-F1':report['VASC']['f1-score'], 'BKL-F1':report['BKL']['f1-score'],
        'NV-F1':report['NV']['f1-score']})
        if classes_n==9:
            wandb.log({'UNK-F1-val':report['UNK']['f1-score']})   
        # =============================================================================
        # ROC AUC
        # =============================================================================
        ROC_AUC = roc_auc_score(y_test,predict[:,:classes_n], multi_class='ovr')
        auc_ak = roc_auc_score((y_test == labels.index('AK')).astype(float), predict[:, labels.index('AK')])
        auc_bcc = roc_auc_score((y_test == labels.index('BCC')).astype(float), predict[:, labels.index('BCC')])
        auc_bkl = roc_auc_score((y_test == labels.index('BKL')).astype(float), predict[:, labels.index('BKL')])
        auc_df = roc_auc_score((y_test == labels.index('DF')).astype(float), predict[:, labels.index('DF')])
        auc_scc = roc_auc_score((y_test == labels.index('SCC')).astype(float), predict[:, labels.index('SCC')])
        auc_vasc = roc_auc_score((y_test == labels.index('VASC')).astype(float), predict[:, labels.index('VASC')])
        auc_mel = roc_auc_score((y_test == labels.index('MEL')).astype(float), predict[:, labels.index('MEL')])
        auc_nv = roc_auc_score((y_test == labels.index('NV')).astype(float), predict[:, labels.index('NV')])
        if classes_n==9:
            auc_unk = roc_auc_score((y_test == labels.index('UNK')).astype(float), predict[:, labels.index('UNK')])
            wandb.log({'val-unk-auc':auc_unk})
    
        wandb.log({'ak-auc':auc_ak,'bkl-auc':auc_bkl,'df-auc':auc_df,'vasc-auc':auc_vasc,'mel-auc':auc_mel,
                    'bcc-auc':auc_bcc , 'scc-auc':auc_scc,'nv-auc':auc_nv})
        wandb.log({'ROC-AUC': ROC_AUC})
        #predict=pd.DataFrame(predict)
        return predict
    else: 
        return get_tta_probs(probs) 

def get_tta_probs(probs,class_n):
    # best tta method: df = df.groupby(df.index // 20).agg({'image_name': 'first', 'y_true': 'first', 'target': 'mean'})
    probs = np.array(probs)
    print(f"shape of  test probs: {probs.shape}")
    names = probs[:,class_n]
    indices_to_keep = np.arange(0, names.shape[0], 20)
    true_names = names[indices_to_keep]
    #print(f"shape of  probs now: {probs[:,:class_n].shape}")
    reshaped_data = probs[:,:class_n].reshape(-1, 20, class_n)
    averaged_data = np.mean(reshaped_data, axis=1)
    probs =np.column_stack((averaged_data,true_names))
    print(f"shape of averages test probs: {probs.shape}")
    return probs

def read_clinical(probs,y_test,classes_n):
        print("this clinical function might need modification if you're using a new dataset")
        print(set(y_test))
        print(probs)
        predict = np.array(probs)
        results = []
        for i in range(len(predict)):
            pred = np.argmax(predict[i][0:classes_n])
            results.append(pred)

        y_pred = results 
        print(set(y_test))
        print(set(y_pred))
        y_test= np.array(y_test)
        y_pred= np.array(y_pred)
        try:
            target_names = ['AK', 'BCC','BKL','DF','SCC','MEL','NV'] #'VASC'
            report= metrics.classification_report(y_test, y_pred, target_names=target_names,output_dict=True)
        except:
            target_names = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV'] 
            report= metrics.classification_report(y_test, y_pred, target_names=target_names,output_dict=True)

        test_f1 = metrics.f1_score(y_test, y_pred, average='weighted')

        # precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        # PR_AUC = auc(recall, precision)
        # print("auc pr:",PR_AUC)

        labels = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV']
        pr_ak , rec_ak , _= precision_recall_curve((y_test == labels.index('AK')).astype(float), predict[:, labels.index('AK')])
        pr_bcc, rec_bcc , _ = precision_recall_curve((y_test == labels.index('BCC')).astype(float), predict[:, labels.index('BCC')])
        pr_bkl, rec_bkl , _= precision_recall_curve((y_test == labels.index('BKL')).astype(float), predict[:, labels.index('BKL')])
        pr_df, rec_df , _ = precision_recall_curve((y_test == labels.index('DF')).astype(float), predict[:, labels.index('DF')])
        pr_scc , rec_scc , _ = precision_recall_curve((y_test == labels.index('SCC')).astype(float), predict[:, labels.index('SCC')])
        pr_vasc , rec_vasc , _= precision_recall_curve((y_test == labels.index('VASC')).astype(float), predict[:, labels.index('VASC')])
        pr_mel , rec_mel , _ = precision_recall_curve((y_test == labels.index('MEL')).astype(float), predict[:, labels.index('MEL')])
        pr_nv , rec_nv , _ = precision_recall_curve((y_test == labels.index('NV')).astype(float), predict[:, labels.index('NV')])

        
        pr_auc_ak =  auc(rec_ak , pr_ak)
        pr_auc_bcc = auc(rec_bcc , pr_bcc)
        pr_auc_bkl= auc(rec_bkl, pr_bkl)
        pr_auc_df = auc(rec_df , pr_df)
        pr_auc_scc = auc(rec_scc , pr_scc)
        pr_auc_vasc = auc(rec_vasc , pr_vasc)
        pr_auc_mel = auc(rec_mel , pr_mel)
        pr_auc_nv = auc(rec_nv , pr_nv)

        unique_samples, sample_counts = np.unique(y_test, return_counts=True)
        weights = class_weights(sample_counts)
        pr_auc = (weights[labels.index('AK')]* pr_auc_ak + weights[labels.index('BCC')]*pr_auc_bcc +
                   weights[labels.index('BKL')]*pr_auc_bkl + weights[labels.index('DF')]*pr_auc_df +
                   weights[labels.index('SCC')]* pr_auc_scc + weights[labels.index('VASC')]*pr_auc_vasc +
                   weights[labels.index('MEL')]*pr_auc_mel + weights[labels.index('NV')]* pr_auc_nv) / classes_n
     
        wandb.log ({'PR-AUC': pr_auc})
        
        wandb.log({'test-f1': test_f1,'test_precision':report['weighted avg']['precision'], 
        'test_recall': report['weighted avg']['recall'], 'Mel-F1': report['MEL']['f1-score'], 'BCC-F1' : report['BCC']['f1-score'], 'AK-F1' :report['AK']['f1-score'] , 
        'SCC-F1':report['SCC']['f1-score'], 'DF-F1':report['DF']['f1-score'], 'BKL-F1':report['BKL']['f1-score'],
        'NV-F1':report['NV']['f1-score']}) #'VASC-F1':report['VASC']['f1-score']
        
        wandb.log({'AK-PR-AUC': pr_auc_ak, 'BCC-PR-AUC': pr_auc_bcc,'BKL-PR-AUC': pr_auc_bkl,'DF-PR-AUC': pr_auc_df,
                   'SCC-PR-AUC': pr_auc_scc,'MEL-PR-AUC': pr_auc_mel,'NV-PR-AUC': pr_auc_nv})
        # =============================================================================
        # ROC AUC
        # =============================================================================
      
        auc_ak = roc_auc_score((y_test == labels.index('AK')).astype(float), predict[:, labels.index('AK')])
        auc_bcc = roc_auc_score((y_test == labels.index('BCC')).astype(float), predict[:, labels.index('BCC')])
        auc_bkl = roc_auc_score((y_test == labels.index('BKL')).astype(float), predict[:, labels.index('BKL')])
        auc_df = roc_auc_score((y_test == labels.index('DF')).astype(float), predict[:, labels.index('DF')])
        auc_scc = roc_auc_score((y_test == labels.index('SCC')).astype(float), predict[:, labels.index('SCC')])
        auc_vasc = roc_auc_score((y_test == labels.index('VASC')).astype(float), predict[:, labels.index('VASC')])
        auc_mel = roc_auc_score((y_test == labels.index('MEL')).astype(float), predict[:, labels.index('MEL')])
        auc_nv = roc_auc_score((y_test == labels.index('NV')).astype(float), predict[:, labels.index('NV')])

        #add 'vasc-auc':auc_vasc
        wandb.log({'ak-auc':auc_ak,'bkl-auc':auc_bkl,'df-auc':auc_df,'mel-auc':auc_mel, 'bcc-auc':auc_bcc , 'scc-auc':auc_scc,'nv-auc':auc_nv})
        #predict=pd.DataFrame(predict)
        return predict

def read_prove(probs,y_test,classes_n):
        print(set(y_test))
        print("len y test",len(y_test))
        
        probs = get_tta_probs(probs,classes_n) 
        indices_to_keep = np.arange(0, y_test.shape[0], 20)
        y_test =  y_test[indices_to_keep]
        print(probs)
        predict = np.array(probs)
        results = []
        for i in range(len(predict)):
            pred = np.argmax(predict[i][0:classes_n])
            results.append(pred)

        y_pred = results 
        print(set(y_pred))
        y_test= np.array(y_test)
        y_pred= np.array(y_pred)

             
        labels = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV']             
        test_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        print("test f1:",test_f1)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        wandb.log({'b_acc':balanced_acc})
        report= metrics.classification_report(y_test, y_pred, target_names=labels,output_dict=True)
        wandb.log({'test-f1': test_f1,'test_precision':report['weighted avg']['precision'], 
        'test_recall': report['weighted avg']['recall'], 'Mel-F1': report['MEL']['f1-score'], 'BCC-F1' : report['BCC']['f1-score'], 'AK-F1' :report['AK']['f1-score'] , 
        'SCC-F1':report['SCC']['f1-score'], 'DF-F1':report['DF']['f1-score'],'VASC-F1':report['VASC']['f1-score'], 'BKL-F1':report['BKL']['f1-score'],
        'NV-F1':report['NV']['f1-score']})
        # =============================================================================
        pr_ak , rec_ak , _= precision_recall_curve((y_test == labels.index('AK')).astype(float), predict[:, labels.index('AK')])
        pr_bcc, rec_bcc , _ = precision_recall_curve((y_test == labels.index('BCC')).astype(float), predict[:, labels.index('BCC')])
        pr_bkl, rec_bkl , _= precision_recall_curve((y_test == labels.index('BKL')).astype(float), predict[:, labels.index('BKL')])
        pr_df, rec_df , _ = precision_recall_curve((y_test == labels.index('DF')).astype(float), predict[:, labels.index('DF')])
        pr_scc , rec_scc , _ = precision_recall_curve((y_test == labels.index('SCC')).astype(float), predict[:, labels.index('SCC')])
        pr_mel , rec_mel , _ = precision_recall_curve((y_test == labels.index('MEL')).astype(float), predict[:, labels.index('MEL')])
        pr_nv , rec_nv , _ = precision_recall_curve((y_test == labels.index('NV')).astype(float), predict[:, labels.index('NV')])
        pr_vasc, rec_vasc  , _= precision_recall_curve((y_test == labels.index('VASC')).astype(float), predict[:, labels.index('VASC')])       
 
        pr_auc_ak =  auc(rec_ak , pr_ak)
        pr_auc_bcc = auc(rec_bcc , pr_bcc)
        pr_auc_bkl= auc(rec_bkl, pr_bkl)
        pr_auc_df = auc(rec_df , pr_df)
        pr_auc_scc = auc(rec_scc , pr_scc)
        pr_auc_mel = auc(rec_mel , pr_mel)
        pr_auc_nv = auc(rec_nv , pr_nv)
        pr_auc_vasc = auc(rec_vasc , pr_vasc)
        #weighted
        unique_samples, sample_counts = np.unique(y_test, return_counts=True)
        weights = class_weights(sample_counts)
        w_pr_auc = (weights[labels.index('AK')]* pr_auc_ak + weights[labels.index('BCC')]*pr_auc_bcc +
                   weights[labels.index('BKL')]*pr_auc_bkl + weights[labels.index('DF')]*pr_auc_df +
                   weights[labels.index('SCC')]* pr_auc_scc + weights[labels.index('VASC')]*pr_auc_vasc +
                   weights[labels.index('MEL')]*pr_auc_mel + weights[labels.index('NV')]* pr_auc_nv) / classes_n
        
        pr_auc = (pr_auc_ak + pr_auc_bcc + pr_auc_bkl + pr_auc_df + pr_auc_scc + pr_auc_vasc + pr_auc_mel + pr_auc_nv) / classes_n
       
        wandb.log ({'W-PR-AUC': w_pr_auc,'PR-AUC': pr_auc})
        wandb.log({'AK-PR-AUC': pr_auc_ak, 'BCC-PR-AUC': pr_auc_bcc,'BKL-PR-AUC': pr_auc_bkl,'DF-PR-AUC': pr_auc_df,
                   'SCC-PR-AUC': pr_auc_scc,'MEL-PR-AUC': pr_auc_mel,'NV-PR-AUC': pr_auc_nv,'VASC-PR-AUC': pr_auc_vasc})
        # =============================================================================
        # ROC AUC
        # =============================================================================
        #labels = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV']
        ROC_AUC = roc_auc_score(y_test,predict[:,:classes_n], multi_class='ovr')
        auc_ak = roc_auc_score((y_test == labels.index('AK')).astype(float), predict[:, labels.index('AK')])
        auc_bcc = roc_auc_score((y_test == labels.index('BCC')).astype(float), predict[:, labels.index('BCC')])
        auc_bkl = roc_auc_score((y_test == labels.index('BKL')).astype(float), predict[:, labels.index('BKL')])
        auc_df = roc_auc_score((y_test == labels.index('DF')).astype(float), predict[:, labels.index('DF')])
        auc_scc = roc_auc_score((y_test == labels.index('SCC')).astype(float), predict[:, labels.index('SCC')])
        auc_mel = roc_auc_score((y_test == labels.index('MEL')).astype(float), predict[:, labels.index('MEL')])
        auc_nv = roc_auc_score((y_test == labels.index('NV')).astype(float), predict[:, labels.index('NV')])
        auc_vasc = roc_auc_score((y_test == labels.index('VASC')).astype(float), predict[:, labels.index('VASC')])
        wandb.log({'ak-auc':auc_ak,'bkl-auc':auc_bkl,'df-auc':auc_df,'mel-auc':auc_mel, 'bcc-auc':auc_bcc , 'scc-auc':auc_scc,'nv-auc':auc_nv,'vasc-auc':auc_vasc})
        wandb.log({'ROC-AUC': ROC_AUC})
        print("ROC AUC:",ROC_AUC)
        #predict=pd.DataFrame(predict)
        return predict

 
      
     