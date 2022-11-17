#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse
import os
import math
import tqdm
import cv2
from PIL import Image
import scipy.sparse
#%%
DATA_DIR = "C:/Users/Owner/Documents/dev/open-problem/open-problems-multimodal"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")


TRAIN = True

# X = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS)
if TRAIN:
    OUTPUT_DIR = "C:/Users/Owner/Documents/dev/open-problem/output/imagedata/multi-minmax/train"
    X = scipy.sparse.load_npz("C:/Users/Owner/Documents/dev/open-problem/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_values.sparse.npz").tocsr()
    idx = np.load("C:/Users/Owner/Documents/dev/open-problem/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_idxcol.npz", allow_pickle=True)["index"]

else:
    OUTPUT_DIR = "C:/Users/Owner/Documents/dev/open-problem/output/imagedata/multi-minmax/test"
    X = scipy.sparse.load_npz("C:/Users/Owner/Documents/dev/open-problem/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_values.sparse.npz").tocsr()
    idx = np.load("C:/Users/Owner/Documents/dev/open-problem/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_idxcol.npz", allow_pickle=True)["index"]
#%%
feat = 228942
# w = math.ceil(math.sqrt(feat))
w = 480
#%%
# X.columns.str.split('_').map(lambda x: x[0])
# cols = pd.DataFrame(X.columns.str.split('_').map(lambda x: x[0]))
# positions = pd.read_csv("C:/Users/Owner/Documents/dev/open-problem/positions.tsv", sep="\t")[["Gene stable ID", "Gene start (bp)"]].drop_duplicates()
# #%%
# positions = pd.merge(cols, positions, left_on="gene_id", right_on="Gene stable ID", how="left")
#%%
# X.max().max()
# X.min(axis=1).reshape(-1, 1)
#%%
# X = X/12.705485 *255*255
# X -= X.min(axis=1).reshape(-1, 1)
# X /= X.max(axis=1).reshape(-1, 1) - X.min(axis=1).reshape(-1, 1) 
# X = X*255*255 
#%%
X.shape[0] == len(idx)
#%%
for i in tqdm.tqdm(range(X.shape[0])):
    row = X[i].toarray().squeeze()
    row -= row.min()
    row /= row.max() - row.min()
    row = row*255*255

    name = idx[i]
    im = np.concatenate([row, np.zeros((w*w-feat))]).reshape([w,w]).astype(np.uint16)
    Image.fromarray(im).convert("I;16").save(os.path.join(OUTPUT_DIR, name+".png"))
#%%
#%%
    # df = pd.DataFrame(row)
    # df.index = df.index.str.split("_").map(lambda x: x[0])
    # onedim = pd.merge(df, positions, left_on="gene_id", right_on="Gene stable ID", how="left").sort_values("Gene start (bp)").iloc[:,0].values
    # im = np.concatenate([onedim, np.zeros((w*w-feat))], axis=0).reshape([w,w]).astype(np.uint16)

    # # cv2.imwrite(os.path.join(OUTPUT_DIR, name+".jpg"), im)
    # Image.fromarray(im).convert("I;16").save(os.path.join(OUTPUT_DIR, name+".png"))