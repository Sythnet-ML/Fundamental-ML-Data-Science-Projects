import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%matplotlib inline

# SKLearn 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline



# ====================================================================
#   DATA PREPARATION
# ====================================================================

# Load Data
def load_data(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(e)
        
    return df


# ====================================================================
#   MODEL BUILDING FUNCTION
# ====================================================================
def metric(y_true, y_pred, model = ''):
    
    metric = {'model': model,
        'accuracy' : accuracy_score(y_true, y_pred),
                'precision' : precision_score(y_true, y_pred),
                'recall' : recall_score(y_true, y_pred),
                'f1_score' : f1_score(y_true, y_pred),
    }
    
    print('='*75)
    print(metric['model'])
    
    print(f'Accuracy \t{(metric['accuracy']* 100):.4f}%')
    print(f'Precision \t{(metric['precision'] * 100):.4f}%')
    print(f'Recall \t\t{(metric['recall'] * 100):.4f}%')
    print(f'F1 Score \t{(metric['f1_score'] * 100):.4f}%')
    
    print('='*75)
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred))
    
    print('='*75)
    return metric

def confusion_matrix_diagram(y_true, y_pred, model=''):
    cm = confusion_matrix(y_true, y_pred)
    print(f"{model} CONFUSION MATRIX")
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, 
                cmap='Blues',
                annot= True,
                square= True,
                cbar=True,)
    
    plt.xlabel("Predicted")
    plt.ylabel('Actual')
    plt.show()