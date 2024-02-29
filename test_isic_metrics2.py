import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_recall_curve, precision_score, recall_score, roc_curve, confusion_matrix
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

directory1 = './ISIC2020_winners/predictions/ensembles-rank/prove'

dfs = []
file_names = []
optimal_thresholds= {}
scores = [["file_name", "roc_auc", "pr_auc", "f1-score", "accuracy"]]
# Plot all ROC curves on the same plot
plt.figure(figsize=(8, 6))

for csv in sorted(glob(os.path.join(directory1, '*csv'))):
    df = pd.read_csv(csv)
    dfs.append(df)
    file_name = os.path.splitext(os.path.basename(csv))[0]  # Remove ".csv" extension
    file_names.append(file_name)

for i in range(len(dfs)):
    df = dfs[i]
    if 'units' in directory1:
        df['y_true'] = df['image_name'].str[0].astype(int).apply(lambda x: 1 if x == 6 else 0)
        #prove
    if len(df) > 604:
        # TTA
        df = df.groupby(df.index // 20).agg({'image_name': 'first', 'y_true': 'first', 'target': 'mean'})

    roc_auc = roc_auc_score(df['y_true'], df['target'])
    (df['y_true'], df['target'])
    fpr, tpr, thresholds = roc_curve(df['y_true'], df['target'])
    #define optimal threshold
    desired_sensitivity = 0.94
    min_specificity = 0.18
    optimal_threshold_index = np.argmax(tpr >= desired_sensitivity)
    optimal_threshold = thresholds[optimal_threshold_index]
    print(f"Optimal Threshold for {file_names[i]}:", optimal_threshold)
    optimal_thresholds[file_names[i]] = optimal_threshold
    plt.plot(fpr, tpr, label=f'{file_names[i]} - AUC: {roc_auc:.2f}')
print(optimal_thresholds)
plt.plot([0, 1], [0, 1], color='r', linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join("./plots/", "all_roc_curves.png"))
print("roc plot saved")


# Plot all precision-recall curves on the same plot
plt.figure(figsize=(8, 6))

for i in range(len(dfs)):
    df = dfs[i]
    pr_auc = average_precision_score(df['y_true'], df['target'], average='weighted')
    precision, recall, _ = precision_recall_curve(df['y_true'], df['target'])
    plt.plot(recall, precision, label=f'{file_names[i]}- PR-AUC: {pr_auc:.2f}')

plt.plot([0, 1], [df['y_true'].sum() / len(df)] * 2, linestyle='--', color='gray', label='No Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig(os.path.join("./plots/", "all_precision_recall_curves.png"))
print("precision-recall plot saved")

# Plot sensitivity-specificity curve for all datasets on the same plot
plt.figure(figsize=(8, 6))

for i in range(len(dfs)):
    df = dfs[i]
    sensitivity = []
    specificity = []

    thresholds = np.linspace(0, 1, 100)

    for threshold in thresholds:
        y_pred = (df['target'] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(df['y_true'], y_pred).ravel()

        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))

    plt.plot(1 - np.array(specificity), sensitivity, label=f'{file_names[i]} - Sensetivity: {sensitivity[i]:.2f}')

plt.title('Sensitivity-Specificity Chart')
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join("./plots/", "all_sensitivity_specificity_curves.png"))
print("sensetivity- specificity plot saved")

# Display or save summary scores
scores_df = pd.DataFrame(scores[1:], columns=scores[0])
print(scores_df)
