import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,  mean_absolute_percentage_error

#===================================================================================
# LOAD DATASET
#===================================================================================
def load_data(path= 'data\\avocado.csv'):
    df = pd.read_csv(path)
    
    return df


#===================================================================================
# Data Cleaning and Feature Engineer 
#===================================================================================
def drop_index(df):
    df = df.drop('Unnamed: 0', axis=1)
    return df

def parse_data(df):
    df_clean = df.copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean['Month'] = df_clean['Date'].dt.month
    df_clean['Week'] = df_clean['Date'].dt.isocalendar().week.astype(int)
    
    return df_clean

def rename_plu(df):
    df_clean = df.copy()
    
    df_clean = df_clean.rename(columns = {
        "Total Volume": "total_volume",
        "4046": "small_hass",
        "4225": "large_hass",
        "4770": "xl_hass",
        "Total Bags": "total_bags",
        "Small Bags": "small_bags",
        "Large Bags": "large_bags",
        "XLarge Bags": "xl_bags",
        "AveragePrice": "AveragePrice",
        "Date": "Date",
        "type": "type",
        "year": "year",
        "region": "region",
    })
    
    return df_clean

#===================================================================================
# Feature Engineering
#===================================================================================

def create_feature(df):
    df_feat = df.copy()
    
    df_feat['log_total_volume'] = np.log1p(df_feat['total_volume'])
    df_feat['log_total_bags'] = np.log1p(df_feat['total_bags'])
    df_feat['log_small_bags'] = np.log1p(df_feat['small_bags'])
    df_feat['log_large_bags'] = np.log1p(df_feat['large_bags'])
    df_feat['log_xl_bags'] = np.log1p(df_feat['xl_bags'])



    t_bags = df_feat[['small_hass','large_hass','xl_hass']].sum(axis=1).replace(0, np.nan)
    df_feat['small_share'] = df_feat['small_hass'] / t_bags
    df_feat['large_share'] = df_feat['large_hass'] / t_bags
    df_feat['xl_share'] = df_feat['xl_hass'] / t_bags

    return df_feat

def preprocessing_data(df):
    df_clean = drop_index(df)
    df_clean = parse_data(df_clean)
    df_clean = rename_plu(df_clean)
    df_clean = create_feature(df_clean)
    df_clean = df_clean.drop(columns='Date')
    
    cat_col = ['type' ,'region']
    df_encoded = pd.get_dummies(data=df_clean ,columns=cat_col, drop_first=True)
    bool_encode = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_encode] = df_encoded[bool_encode].astype(int)
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only= True)) 
    
    return df_encoded


#===================================================================================
# Model Function
#===================================================================================


def metric(y_true, y_pred, name ='S'):
    metric = {
        'Model' : name,
        'RMSE' : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2' : r2_score(y_true, y_pred),
        'MAE' : mean_absolute_error(y_true, y_pred),
        'MAPE' : mean_absolute_percentage_error(y_true, y_pred),
    }
    
    print(f"\n{'='*45}")
    print(f"     {name}")
    print(f"{'='*45}")
    
    for k, v in metric.items():
        if k != "Model":
            print(f"    {k:.6s}: {v:.4f}")
    return metric    

def plot_a_vs_pred(y_true, y_pred, model_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize = (7,5))
    
    ax.scatter(y_true, y_pred, alpha= 0.3, s= 10)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'go--', linewidth=0.5, label= 'Perfect')
    ax.set_xlabel("Actual");    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual Vs Predicted - {model_name}")
    ax.legend()
    
    return ax

def plot_residual(y_true, y_pred, model_name, ax= None):
    residual = y_true - y_pred
    
    if ax is None:
        fig, ax = plt.subplots(figsize= (7,5))
        
    ax.scatter(y_pred, residual, alpha=0.3, color='salmon', s=10)
    ax.axhline(0, color= 'black', linestyle= '--', linewidth= 1)
    ax.set_xlabel("Predictor");     ax.set_ylabel("Residual")
    ax.set_title(f"Residual - {model_name}")
    
    return ax

