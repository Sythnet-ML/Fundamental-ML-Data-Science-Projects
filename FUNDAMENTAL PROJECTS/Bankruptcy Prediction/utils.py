import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score,precision_score
                            , recall_score, f1_score, confusion_matrix,
                            roc_curve, auc, classification_report)

sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6)})

import warnings

warnings.filterwarnings("ignore")



# ------------------------------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------------------------------

def load_data(path= 'data\\bankrupt.csv'):
    df = pd.read_csv(path)
    df['Bankrupt'] = df['Bankrupt'].map({'Yes': 1, 'No': 0})
    
    return df

# ------------------------------------------------------------------------------------------
# Model Function
# ------------------------------------------------------------------------------------------
target_label = ['Healthy', 'Bankrupt']

def evaluate_metric(y_test ,y_pred, model_name):
    metric = {
        'Model' : model_name,
        'Accuracy' : accuracy_score(y_test, y_pred),
        'Precision' : precision_score(y_test, y_pred, zero_division=0),
        'Recall' : recall_score(y_test, y_pred, zero_division=0),
        'F1 Score' : f1_score(y_test, y_pred, zero_division=0),
    }
    
    print(f'{'='* 75}')
    print(f"Metric of {model_name}:\n")
    
    
    print(f"{'Accuracy':<10} {(metric['Accuracy'] * 100):.3f}%")
    print(f"{'Precision':<10} {(metric['Precision']* 100):.3f}%")
    print(f'{'Recall':<10} {(metric['Recall']* 100):.3f}%')
    print(f"{"F1 Score":<10} {(metric['F1 Score']* 100):.3f}%")
    
    print(f'{'='* 75}')
    
    return metric

def plot(y_true, y_pred, model_name):
    fig, axes = plt.subplots(figsize=(6, 4))
    ## Confusion Matrix Plot - Heatmap
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm , annot=True, cmap='Reds',fmt= 'd', ax= axes, xticklabels=target_label, yticklabels=target_label)
    axes.set_title(f"Confusion Matrix -- {model_name}")
    
    plt.tight_layout(); plt.show()
    
def cross_val()

