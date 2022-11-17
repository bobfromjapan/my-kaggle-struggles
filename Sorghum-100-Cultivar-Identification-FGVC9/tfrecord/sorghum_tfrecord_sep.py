#%%
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import tensorflow as tf
import dill
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import glob
import os
#%%
import tensorflow_datasets as tfds
#%%
# proj_dir = "/home/bob/dev/sorghum/data"
proj_dir = "C:/Users/Owner/Documents/dev/sorghum/data/"

train_df = pd.read_csv("512_separated_clahe_train_cultivar_mapping.csv")
# train_df = pd.read_csv("/home/bob/dev/container/128_separated_train_cultivar_mapping.csv")
# test_df = pd.read_csv("C:/Users/Owner/Documents/dev/sorghum/0_dataset/separated_clahe_256_test_cultivar_mapping.csv")
# test_df = pd.read_csv("/home/bob/dev/container/128_separated_test_cultivar_mapping.csv")
test_data_paths = glob.glob(proj_dir + 'sorghum-id-fgvc-9/test_images_clahe_512x512/*.jpg')
test_data_paths

#%%
base_path = proj_dir + "sorghum-id-fgvc-9/train_images_clahe_512x512/"

# train_df["fullpath"] = [base_path + os.path.splitext(os.path.basename(name))[0] + ".jpg" for name in train_df["image"]]
# test_data_paths = test_df["fullpath"]
#%%
# train_df["fullpath"][9]
#%%
# test_df = pd.DataFrame({
#     'image':[os.path.basename(name) for name in glob.glob(test_data_paths)],
#     'fullpath':[name for name in glob.glob(test_data_paths)]
# })
test_df = pd.DataFrame({
    'image':[os.path.splitext(os.path.basename(name))[0] + ".jpg" for name in test_data_paths],
    'fullpath':[name for name in test_data_paths]
})
test_df['cultivar'] = ''
test_df.to_csv("submission_clahe.csv",index=False)
#%%
# exists = []

# for i in train_df["fullpath"]:
#     if not os.path.exists(i):
#         exists.append(False)
#         print(i)
#     else:
#         exists.append(True)

# train_df["exist"] = pd.Series(exists)
# train_df = train_df[train_df.exist]
# len(train_df)
#%%
paths = train_df["fullpath"]
labels_str = train_df["cultivar"]
label_to_index = dict((name, index) for index,name in enumerate(labels_str.unique()))
# label_to_index
labels_idx = labels_str.map(lambda x: label_to_index[x])

train_df["target"] = labels_idx

#%%
# train_df["image"] = pd.Series([os.path.splitext(os.path.basename(name))[0] + ".png" for name in train_df["fullpath_org"]])
#%%
# train_df.iloc[1].fullpath
#%%
test_df
#%%
# import tensorflow_datasets as tfds
# import tensorflow as tf

# class FGVCDataset(tfds.core.GeneratorBasedBuilder):
#     VERSION = tfds.core.Version('0.1.0')
    
#     def _split_generators(self, dl_manager):
#         arr = [
#             tfds.core.SplitGenerator(name=f'train',gen_kwargs={"split":"train"}),
#             tfds.core.SplitGenerator(name=f'test',gen_kwargs={"split":"test"})
#         ]
#         return arr
    
#     def _info(self):
#         return tfds.core.DatasetInfo(
#             builder=self,
#             description=(""),
#             #disable_shuffling=True,
#             features=tfds.features.FeaturesDict({
#                 "img": tfds.features.Image(encoding_format='png'),#dtype=tf.uint8,shape=(self.WIDTH,self.HEIGHT,3),
#                 "name": tfds.features.Tensor(dtype=tf.string,shape=()),
#                 "cultivar": tfds.features.Tensor(dtype=tf.string,shape=()),
#                 "target": tfds.features.Tensor(dtype=tf.int32,shape=()),
#             }),
#         )
    
#     def _generate_examples(self,**args):
#         print(args)
#         split = args["split"]
        
#         if split == 'train':
#             for i in range(len(self.train_df)):
#                 row = self.train_df.iloc[i]
#                 img = row.fullpath
#                 yield i, {
#                     'img':img,
#                     'cultivar':row.cultivar,
#                     'name':row.image,
#                     'target':row.target,
#                 }
                
#         if split == 'test':
#             for i in range(len(self.test_df)):
#                 row = self.test_df.iloc[i]
#                 img = row.fullpath
#                 yield i, {
#                     'img':img,
#                     'name':row.image,
#                     'cultivar':'',
#                     'target':-1,
#                 }

#%%
from fgvc_dataset import FGVCDataset

data_dir='fgvc_dataset_clahe_512x512'
FGVCDataset.train_df = train_df#.iloc[:100]
FGVCDataset.test_df = test_df#.iloc[:100]

builder = FGVCDataset(data_dir=data_dir)
builder.download_and_prepare() 

#%%
builder.as_dataset()

#%%
train_ds = builder.as_dataset()['train']
test_ds = builder.as_dataset()['test']

#%%
for x in train_ds.take(1):
    img = x['img']
    cultivar = x['cultivar']
    target = x['target']
    print(img.shape,cultivar,target)

#%%
# plt.imshow(img.numpy())
train_df.to_csv("train_cultivar_mapping_clahe_512.csv")