#%%
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# proj_dir = "/home/bob/dev/sorghum/"
proj_dir = "C:/Users/Owner/Documents/dev/sorghum/"

df = pd.read_csv(proj_dir + "train_cultivar_mapping.csv")
#%%
df.groupby("cultivar").describe().to_csv(proj_dir + "cultivars.csv")

#%%
base_path = proj_dir + "sorghum-id-fgvc-9/train_images/"

df["fullpath"] = base_path + df["image"] 

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
BATCH_SIZE = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_SIZE = 0.9
SHUFFLE_SIZE = 5000
test_data_paths = proj_dir + 'sorghum-id-fgvc-9/test/*.png'
WIDTH = 256
HEIGHT = 256
EPOCHS = 30
model_name = "simple_effnet"
# PREPROCESS = tf.keras.applications.efficientnet_v2.preprocess_input
PREPROCESS = tf.keras.applications.efficientnet.preprocess_input

paths = df["fullpath"]
labels_str = df["cultivar"]


image_count = len(paths)
image_count
#%%
label_to_index = dict((name, index) for index,name in enumerate(labels_str.unique()))
label_to_index
#%%
# len(label_to_index)
#%%
labels_idx = labels_str.map(lambda x: label_to_index[x])
# labels_idx

#%%
train_paths, val_paths, train_labels, val_labels = train_test_split(paths, labels_idx, train_size=TRAIN_SIZE, shuffle=True, random_state=42, stratify=labels_idx)
#%%
#%%
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8,8))
# for n,image in enumerate(path_ds.take(4)):
#     print(image)
#     plt.subplot(2,2,n+1)
#     plt.imshow(tf.image.decode_png(tf.io.read_file(image)))
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()

#%%
# def preprocess_image(image):
#     image = tf.image.resize(image, [WIDTH, HEIGHT])
#     # image /= 255.0  # normalize to [0,1] range
#     # image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

#     return image


def preprocess_image(path):
    image = tf.image.decode_png(path, channels=3)
    image = tf.image.resize(image, size=(HEIGHT,WIDTH))

    image = PREPROCESS(image)

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

def data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
data_aug = data_augmentation()
#%%
train_path_label_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_path_label_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

#%%
train_image_label_ds = train_path_label_ds.map(load_and_preprocess_from_path_label,num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_image_label_ds = val_path_label_ds.map(load_and_preprocess_from_path_label,num_parallel_calls=tf.data.experimental.AUTOTUNE)

#%%
# for im, lab in image_label_ds.take(4):
#     print(im)
#%%
train_ds = train_image_label_ds
train_ds = train_ds.repeat()
train_ds = train_ds.shuffle(buffer_size=SHUFFLE_SIZE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(lambda x,y:(data_aug(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
train_ds

# val_ds = val_image_label_ds.shuffle(buffer_size = len(val_paths))
# val_ds = val_ds.repeat()
val_ds = val_image_label_ds.cache()
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
val_ds

#%%
# basemodel = tf.keras.applications.MobileNetV2(input_shape=(WIDTH, HEIGHT, 3), weights='imagenet', include_top=False)

#%%
# basemodel = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(input_shape=(WIDTH, HEIGHT, 3), weights='imagenet', include_top=False)
# basemodel.trainable=False

basemodel = tf.keras.applications.EfficientNetB4(include_top=False,weights="imagenet",input_shape=(WIDTH,HEIGHT,3))

#%%
# image_batch, label_batch = next(iter(ds))
# for im, lab in ds.take(4):
#     print(im)

#%%
# model = tf.keras.Sequential([
#     mobile_net,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(len(label_to_index), activation="softmax")])

# model = tf.keras.Sequential([
#     effnet(),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(len(label_to_index), activation="softmax")])

image_input = tf.keras.layers.Input(shape=(WIDTH,HEIGHT,3))
out = basemodel(image_input)
out = tf.keras.layers.GlobalAveragePooling2D()(out)
# out = tf.keras.layers.Dropout(0.2)(out)
out = tf.keras.layers.Dense(len(label_to_index), activation="softmax")(out)

model = tf.keras.Model(image_input, out)

model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])


#%%
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy',mode='max', patience=20)
ckp_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=f'model.h5',
                                            save_weights_only=True,
                                            monitor='accuracy',
                                            mode='max',
                                            options=tf.train.CheckpointOptions(experimental_io_device='/job:localhost'),
                                            save_best_only=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy',mode='max',factor=0.2,patience=3, min_lr=1e-5)
callbacks=[callback,ckp_callback,reduce_lr]


#%%
model.summary()

#%%
steps_per_epoch=tf.math.ceil(len(train_paths)/BATCH_SIZE).numpy()
steps_per_epoch

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
#%%
model.save(proj_dir + 'saved_model/' + model_name)

#%%
# import matplotlib.pyplot as plt
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(EPOCHS)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

#%% 
import glob
test_image_paths = list(glob.glob(test_data_paths))
test_image_paths = [str(path) for path in test_image_paths]
len(test_image_paths)
#%%
path_label_test = tf.data.Dataset.from_tensor_slices(test_image_paths)
#%%
test_ds = path_label_test.map(load_and_preprocess_image, tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
#%%
predictions = model.predict(test_ds, batch_size=BATCH_SIZE)

#%%
prediction_output = [("filename", "cultivar")]

for i in range(len(predictions.argmax(axis=1))):
    pred_idx = predictions.argmax(axis=1)[i]
    prediction_output.append((os.path.basename(test_image_paths[i]), [k for k, v in label_to_index.items() if v == pred_idx][0]))
#%%
pd.DataFrame(prediction_output).to_csv("submission.csv", index=None, header=None)