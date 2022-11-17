#%%
# prepare 256x256 separated images.
import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
# proj_dir = "/home/bob/dev/sorghum/"
proj_dir = "C:/Users/Owner/Documents/dev/sorghum/data/"
# proj_dir = "/kaggle/input/sorghum-id-fgvc-9/"

image_size = [1024, 1024]

df = pd.read_csv("clahe_train_cultivar_mapping.csv")
#%%
base_path = proj_dir + "sorghum-id-fgvc-9/train_images_clahe/"
OUTPUT_PATH =   "C:/Users/Owner/Documents/dev/sorghum/data/sorghum-id-fgvc-9/train_images_clahe_256x256/"

# df["fullpath"] = base_path + df["image"] 

exists = []

for i in df["fullpath"]:
    if not os.path.exists(i):
        exists.append(False)
        print(i)
    else:
        exists.append(True)

#%%
df["exist"] = pd.Series(exists)
#%%
df = df[df.exist]
len(df)
#%%
def load_and_crop(path, n, img_size, label):
    metadata = []
    base_name = os.path.splitext(os.path.basename(path))[0]
    x0 = int(img_size[0]/n)
    y0 = int(img_size[1]/n)

    image = np.array(cv2.imread(path))
    image_cropped = [image[x0*x:x0*(x+1), y0*y:y0*(y+1)] for x in range(n) for y in range(n)]

    for i, im in enumerate(image_cropped):
        output_name = base_name + "_" + str(i) + ".jpg"
        saved_path = os.path.join(OUTPUT_PATH, output_name)
#         print(saved_path)

        ret = cv2.imwrite(saved_path, im,  [cv2.IMWRITE_JPEG_QUALITY, 100])
        if ret == False:
            print("failed imwrite!!")
        metadata.append((path, saved_path, label))

    return pd.DataFrame(metadata, columns=["fullpath_org", "fullpath", "cultivar"])

#%%
output_df = pd.DataFrame()
for i in tqdm(range(len(df))):
    output_df = pd.concat([output_df, load_and_crop(df["fullpath"].iloc[i], 4, image_size, df["cultivar"].iloc[i])])
#%%
# fullpath_org, fullpath, cultivar
output_df.to_csv("separated_clahe_256_train_cultivar_mapping.csv")