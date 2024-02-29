import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, precision_score, recall_score
from glob import glob
from sklearn.utils import resample
import scipy.stats as stats

directory1 = './ISIC2020_winners/predictions/ensembles-rank/prove'

dfs = []
file_names = []
optimal_thresholds = {}
scores = [["file_name", "roc_auc", "pr_auc", "f1-score", "accuracy","sensitivity",
            "specificity",
           "ci_sensitivity", "ci_specificity", "sensitivity std", "specificity std"]]
sensitivity_values = []
specificity_values = []
# Number of bootstrap samples
n_bootstrap = 1000

def cal(y_true, y_scores,n_iterations=n_bootstrap):
    bootstrapped_specificities = []
    bootstrapped_sensitivities = []
    for i in range(n_iterations):
        # Perform bootstrap resampling
        boot_true, boot_scores = resample(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(boot_true, boot_scores)
        # Find the threshold where sensitivity (recall) is 0.95 or higher
        idx = np.where(tpr >= 0.95)[0]
        # Choose the threshold with the highest specificity
        best_idx = np.argmax(1 - fpr[idx])
        best_specificity = 1 - fpr[idx][best_idx]
        best_sensitivity = tpr[idx][best_idx]
        bootstrapped_specificities.append(best_specificity)
        bootstrapped_sensitivities.append(best_sensitivity)       
    # Compute the mean and standard deviation of the bootstrapped specificities
    mean_specificity = np.mean(bootstrapped_specificities)
    std_specificity = np.std(bootstrapped_specificities)
    mean_sensitivity = np.mean(bootstrapped_sensitivities)
    std_sensitivity = np.std(bootstrapped_sensitivities)
    ci_sensitivity = stats.norm.interval(0.95, loc=mean_sensitivity, scale=std_sensitivity)
    ci_specificity = stats.norm.interval(0.95, loc=mean_specificity, scale=std_specificity)

    print(f'Mean sensitivity: {mean_sensitivity}, 95% CI: {ci_sensitivity}')
    print(f'Mean specificity: {mean_specificity}, 95% CI: {ci_specificity}')

    return mean_sensitivity, mean_specificity, std_sensitivity, std_specificity,ci_sensitivity,ci_specificity

for csv in sorted(glob(os.path.join(directory1, '*csv'))):
    dfs.append(pd.read_csv(csv))
    file_name = os.path.splitext(os.path.basename(csv))[0]  # Remove ".csv" extension
    file_names.append(file_name)

for i in range(len(dfs)):
    df = dfs[i]
    print(f"Processing {file_names[i]}...")
    if 'units' in directory1:
        df['y_true'] = df['image_name'].str[0].astype(int).apply(lambda x: 1 if x == 6 else 0)
    if len(df) > 604:
        # TTA
        df = df.groupby(df.index // 20).agg({'image_name': 'first', 'y_true': 'first', 'target': 'mean'})

    roc_auc = roc_auc_score(df['y_true'], df['target'])
    fpr, tpr, thresholds = roc_curve(df['y_true'], df['target'])


    sensitivity, specificity, sensitivity_std, specificity_std, ci_sensitivity,  ci_specificity= cal(df['y_true'], df['target'])
    ############################

    pr_auc = average_precision_score(df['y_true'], df['target'], average='weighted')
    threshold = 0.5
    binary_predictions = np.where(df['target'] >= threshold, 1, 0)
    f1 = f1_score(df['y_true'], binary_predictions, average='weighted')
    accuracy = accuracy_score(df['y_true'], binary_predictions)



    scores.append([file_names[i], roc_auc, pr_auc, f1, accuracy, sensitivity, 
                   specificity,ci_sensitivity,  ci_specificity, sensitivity_std, specificity_std])

    df = pd.DataFrame(scores)
    df.to_csv(f"scores.csv", index=False)

print("Scores:")
print(df)
