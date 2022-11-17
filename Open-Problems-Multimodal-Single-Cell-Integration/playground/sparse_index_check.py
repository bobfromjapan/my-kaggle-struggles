#%%
import numpy as np
import pandas as pd
import scipy.sparse
import os

DATA_DIR = "C:/Users/Owner/Documents/dev/open-problem/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

train_idx = np.load("C:/Users/Owner/Documents/dev/open-problem/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_idxcol.npz", allow_pickle=True)["index"]
test_idx = np.load("C:/Users/Owner/Documents/dev/open-problem/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_idxcol.npz", allow_pickle=True)["index"]
metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology=="multiome"]
metadata_df.shape
#%%
cell_index = metadata_df[:105942].index
meta = metadata_df.reindex(cell_index)
cell_index_test = metadata_df[105942:].index
meta_test = metadata_df.reindex(cell_index_test)
#%%
sum(cell_index == "56390cf1b95e")
#%%