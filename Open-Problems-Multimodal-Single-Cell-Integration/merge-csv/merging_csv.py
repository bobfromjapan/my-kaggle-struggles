#%%
import pandas as pd
import numpy as np
import os
import tqdm
import gc

DATA_DIR = "C:/Users/Owner/Documents/dev/open-problem/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")
SUMPLE_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")

#%%
csvs = {"C:/Users/Owner/Documents/dev/open-problem/multiome/citeseq_svd256_wdo+catboost.csv": "multi",
        "C:/Users/Owner/Documents/dev/open-problem/Multiome-w-Sparse-M-tSVD-32/submission.csv": "multi",
        "C:/Users/Owner/Documents/dev/open-problem/image/submission_effnet+catboost.csv": "cite",
        "C:/Users/Owner/Documents/dev/open-problem/citeseq/submission_svd256_wdo.csv": "cite",
        }

cite_len = 48663*140 #6,812,820
cite_count = 0
cite = np.zeros((48663, 140), dtype=np.float32)

multi_len = 65744180 - 48663*140 #58,931,360
multi_count = 0
multi = np.zeros((16780, 3512), dtype=np.float32)

for i in csvs:
    df = pd.read_csv(i, index_col='row_id', squeeze=True)
    if csvs[i] == "cite":
        cite_pre = df[:cite_len].values.reshape([48663, 140])
        cite_pre -= cite_pre.mean(axis=1).reshape(-1, 1)
        cite_pre /= cite_pre.std(axis=1).reshape(-1, 1)
        cite += cite_pre
        cite_count += 1
        del cite_pre

    elif csvs[i] == "multi":
        multi_pre = df[cite_len:].values.reshape([16780, 3512])
        multi_pre -= multi_pre.mean(axis=1).reshape(-1, 1)
        multi_pre /= multi_pre.std(axis=1).reshape(-1, 1)
        multi += multi_pre
        multi_count += 1
        del multi_pre

    else:
        print("err!")

    gc.collect()

cite = cite/cite_count
multi = multi/multi_count 
# #%%
# df = pd.read_csv(i, index_col='row_id', squeeze=True)
# df.shape
# #%%
# cite_pre = df[:cite_len].values.reshape([48663, 140])
# cite_pre
# #%%
# cite_pre -= cite_pre.mean(axis=1).reshape(-1, 1)
# #%%
# cite_pre.std(axis=1).reshape(-1, 1)[8000:]
#%%
#%%
submission = pd.read_csv(SUMPLE_SUBMISSION, index_col='row_id', squeeze=True)

submission.iloc[:cite_len] = cite.ravel()
submission.iloc[cite_len:] = multi.ravel()
submission.to_csv("merged_submission.csv")

#%%
# metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
# metadata_df = metadata_df[metadata_df.technology=="multiome"]
# metadata_df = metadata_df[metadata_df.day==10]
# metadata_df.index
# #%%
# eval_ids = pd.read_csv("C:/Users/Owner/Documents/dev/open-problem/open-problems-multimodal/evaluation_ids.csv")
# eval_ids.head()
# #%%
# eval_ids.head()
# #%%
# multiomes = pd.DataFrame()
# for ind, i in tqdm.tqdm(enumerate(metadata_df.index)):
#     if len(multiomes) == 0:
#         multiomes = eval_ids[eval_ids["cell_id"] == i]
#     else:
#         multiomes = pd.DataFrame(multiomes, eval_ids[eval_ids["cell_id"] == i])

#     if ind == 100:
#         break
# #%%
# eval_ids[eval_ids["cell_id"] == "b847ba21f59f"]
# #%%
# sum(eval_ids.groupby("cell_id").count()["row_id"] == 3512)