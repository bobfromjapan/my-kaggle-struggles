#%%
import numpy as np
import pandas as pd

def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    
    It is assumed that the predictions are not constant.
    
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)

y_true = np.array([[1,2,3,4,5,6,7,8,9], [9,8,7,6,5,4,3,2,1]]).astype(np.float32)
y_pred = y_true
correlation_score(y_true, y_pred)
#%%
y_pred -= y_pred.mean(axis=1).reshape(-1, 1)
y_pred /= y_pred.std(axis=1).reshape(-1, 1)
#%%
correlation_score(y_true, y_pred)