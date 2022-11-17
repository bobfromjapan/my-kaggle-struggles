#%%
import os, gc, pickle, datetime, scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import tensorflow_io as tfio
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout
from tensorflow.keras.utils import plot_model
# from tensorflow.keras import mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

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

TRAIN_BASEPATH = "C:/Users/Owner/Documents/dev/open-problem/output/imagedata/cite-minmax/train/"
TEST_BASEPATH = "C:/Users/Owner/Documents/dev/open-problem/output/imagedata/cite-minmax/test/"

TUNE = False
SUBMIT = True

submission_name = "submission_effnetv2+catboost.csv"

#%%
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

def negative_correlation_loss(y_true, y_pred):
    """Negative correlation loss function for Keras
    
    Precondition:
    y_true.mean(axis=1) == 0
    y_true.std(axis=1) == 1
    
    Returns:
    -1 = perfect positive correlation
    1 = totally negative correlation
    """
    my = K.mean(tf.convert_to_tensor(y_pred), axis=1)
    my = tf.tile(tf.expand_dims(my, axis=1), (1, y_true.shape[1]))
    ym = y_pred - my
    r_num = K.sum(tf.multiply(y_true, ym), axis=1)
    r_den = tf.sqrt(K.sum(K.square(ym), axis=1) * float(y_true.shape[-1]))
    r = tf.reduce_mean(r_num / r_den)
    return - r

#%%
metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
metadata_df = metadata_df[metadata_df.technology=="citeseq"]
metadata_df.shape

#%%
cell_index = pd.read_hdf(FP_CITE_TRAIN_INPUTS).index
meta = metadata_df.reindex(cell_index)
gc.collect()
cell_index_test = pd.read_hdf(FP_CITE_TEST_INPUTS).index
meta_test = metadata_df.reindex(cell_index_test)
gc.collect()
#%%
Y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
y_columns = list(Y.columns)
Y = Y.values

# Normalize the targets row-wise: This doesn't change the correlations,
# and negative_correlation_loss depends on it
Y -= Y.mean(axis=1).reshape(-1, 1)
Y /= Y.std(axis=1).reshape(-1, 1)
    
print(f"Y shape: {str(Y.shape):14} {Y.size*4/1024/1024/1024:2.3f} GByte")

#%%
LR_START = 0.01
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
SHUFFLE_SIZE = 10000
WIDTH = 160
HEIGHT = 160
EPOCHS = 50
OUTPUT_LEN = Y.shape[1]

def preprocess_image(image):
#     image = tfio.experimental.image.decode_tiff(path)[...,:3]
    image = tf.io.decode_png(image, channels=1, dtype=tf.dtypes.uint16)
    image = tf.image.resize(image, size=(WIDTH,HEIGHT))
    image = tf.broadcast_to(image, (image.shape[0], image.shape[1], 3))
    image = tf.reshape(image, shape=[WIDTH,HEIGHT,3])

    # image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

def load_and_preprocess_from_path(path):
    return load_and_preprocess_image(path)

def my_model():
    """Sequential neural network
    
    Returns a compiled instance of tensorflow.keras.models.Model.
    """
    basemodel = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_shape=(WIDTH, HEIGHT, 3), weights='imagenet', include_top=False, include_preprocessing=False)
    # basemodel = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(WIDTH, HEIGHT, 3), weights='imagenet', include_top=False)

    image_input = tf.keras.layers.Input(shape=(WIDTH,HEIGHT,3))
    out = basemodel(image_input)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dropout(0.3)(out)
    out = tf.keras.layers.Dense(OUTPUT_LEN, activation=None, kernel_regularizer=tf.keras.regularizers.l2(1e-10))(out)

    model = tf.keras.Model(image_input, out)

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=negative_correlation_loss,
                metrics=[negative_correlation_loss])
    
    return model

my_model().summary()

#%%
# Cross-validation
VERBOSE = 2 # set to 2 for more output, set to 0 for less output
N_SPLITS = 3

np.random.seed(1)
tf.random.set_seed(1)

kf = GroupKFold(n_splits=N_SPLITS)

score_list = []
train_paths = TRAIN_BASEPATH + cell_index + ".png"
histories = []

for fold, (idx_tr, idx_va) in enumerate(kf.split(train_paths, groups=meta.donor)):
    start_time = datetime.datetime.now()
    model = None
    gc.collect()

    X_tr = train_paths[idx_tr]
    y_tr = Y[idx_tr]
    X_va = train_paths[idx_va]
    y_va = Y[idx_va]


    train_path_label_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    train_image_label_ds = train_path_label_ds.map(load_and_preprocess_from_path_label,num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=SHUFFLE_SIZE).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    val_path_label_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va))
    val_image_label_ds = val_path_label_ds.map(load_and_preprocess_from_path_label,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, 
                           patience=4, verbose=VERBOSE)
    es = EarlyStopping(monitor="val_loss",
                       patience=12, 
                       verbose=0,
                       mode="min", 
                       restore_best_weights=True)
    callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]

    # Construct and compile the model
    model = my_model()

    # Train the model
    history = model.fit(train_image_label_ds, 
                        validation_data=val_image_label_ds, 
                        epochs=EPOCHS,
                        steps_per_epoch=len(X_tr)//BATCH_SIZE,
                        # verbose=VERBOSE,
                        callbacks=callbacks)
    # del X_tr, y_tr
    
    if SUBMIT:
        model.save(f"model/model_{fold}")
    history = history.history
    histories.append(history)
    callbacks, lr = None, None
    
    # We validate the model
    y_va_pred = model.predict(tf.data.Dataset.from_tensor_slices(X_va).map(load_and_preprocess_from_path,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE))
    corrscore = correlation_score(y_va, y_va_pred)

    print(f"Fold {fold}: {es.stopped_epoch:3} epochs, corr =  {corrscore:.5f}")
    del es, X_va, X_tr#, y_va, y_va_pred
    score_list.append(corrscore)

# Show overall score
print(f"{Fore.GREEN}{Style.BRIGHT}Average  corr = {np.array(score_list).mean():.5f}{Style.RESET_ALL}")

#%%
for hist in histories:
    print(hist)

#%%
corr_list = []
for i in range(len(y_va)):
    corr_list.append(np.corrcoef(y_va[i], y_va_pred[i])[1, 0])
plt.figure(figsize=(10, 4))
plt.hist(corr_list, bins=100, density=True, color='lightgreen')
plt.title('Distribution of correlations')
plt.xlabel('Correlation')
plt.ylabel('Density')
plt.show()

#%%
if True:
    test_pred = np.zeros((len(cell_index_test), 140), dtype=np.float32)
    for fold in range(N_SPLITS):
        print(f"Predicting with fold {fold}")
        Xt = tf.data.Dataset.from_tensor_slices(TEST_BASEPATH + cell_index_test + ".png")
        Xt = Xt.map(load_and_preprocess_from_path,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
        model = load_model(f"model/model_{fold}",
                           custom_objects={'negative_correlation_loss': negative_correlation_loss})
        test_pred += model.predict(Xt)
#     fold = 0
#     print(f"Predicting with fold {fold}")
#     Xt = tf.data.Dataset.from_tensor_slices(TEST_BASEPATH + cell_index_test + ".png")
#     Xt = Xt.map(load_and_preprocess_from_path,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
#     model = load_model(f"model/model_{fold}",
#                        custom_objects={'negative_correlation_loss': negative_correlation_loss})
#     test_pred += model.predict(Xt)
    

    #with open("../input/msci-multiome-quickstart/partial_submission_multi.pickle", 'rb') as f: submission = pickle.load(f)
    submission = pd.read_csv("C:/Users/Owner/Documents/dev/open-problem/multiome/citeseq_svd256_wdo+catboost.csv",
                             index_col='row_id', squeeze=True)
    submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
    assert not submission.isna().any()
    submission.to_csv(submission_name)
    display(submission)