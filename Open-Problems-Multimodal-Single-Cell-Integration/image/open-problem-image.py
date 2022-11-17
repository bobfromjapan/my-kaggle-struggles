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

OUTPUT_DIR = "C:/Users/Owner/Documents/dev/open-problem/output/imagedata/cite-minmax/train"

X = pd.read_hdf(FP_CITE_TRAIN_INPUTS)
# X = pd.read_hdf(FP_CITE_TEST_INPUTS)
#%%
feat = 22050
# w = math.ceil(math.sqrt(feat))
w = 160
#%%
# X.columns.str.split('_').map(lambda x: x[0])
cols = pd.DataFrame(X.columns.str.split('_').map(lambda x: x[0]))
positions = pd.read_csv("C:/Users/Owner/Documents/dev/open-problem/positions.tsv", sep="\t")[["Gene stable ID", "Gene start (bp)"]].drop_duplicates()
#%%
positions = pd.merge(cols, positions, left_on="gene_id", right_on="Gene stable ID", how="left")
#%%
# X.max().max()
#%%
# X = X/12.705485 *255*255
X -= X.values.min(axis=1).reshape(-1, 1)
X /= X.values.max(axis=1).reshape(-1, 1) - X.values.min(axis=1).reshape(-1, 1) 
X = X*255*255 
#%%
for i, (name, row) in tqdm.tqdm(enumerate(X.iterrows())):
    df = pd.DataFrame(row)
    df.index = df.index.str.split("_").map(lambda x: x[0])
    onedim = pd.merge(df, positions, left_on="gene_id", right_on="Gene stable ID", how="left").sort_values("Gene start (bp)").iloc[:,0].values
    im = np.concatenate([onedim, np.zeros((w*w-feat))], axis=0).reshape([w,w]).astype(np.uint16)

    # cv2.imwrite(os.path.join(OUTPUT_DIR, name+".jpg"), im)
    Image.fromarray(im).convert("I;16").save(os.path.join(OUTPUT_DIR, name+".png"))