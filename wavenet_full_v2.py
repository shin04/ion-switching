import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Dense, Add, Multiply, BatchNormalization, Activation, Dropout
import pandas as pd
import numpy as np
import random
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers
import tensorflow_addons as tfa
import gc
from tqdm import tqdm
from scipy import signal

from collections import Counter, defaultdict
import random

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

import os
for dirname, _, filenames in os.walk('/data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# configurations and main hyperparammeters
EPOCHS = 110
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 4000
SEED = 321
LR = 0.001
SPLITS = 5

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

# read data
def read_data():

    train = pd.read_csv('./data/train_clean_kalman.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('./data/test_clean_kalman.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('./data/sample_submission.csv', dtype={'time': np.float32})
    
    Y_train_proba = np.load("./data/Y_train_proba.npy")
    Y_test_proba = np.load("./data/Y_test_proba.npy")
    
    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]

    return train, test, sub

def batching_10(train, test):
    # concatenate data
    batch = 50
    total_batches = 14
    train['set'] = 'train'
    test['set'] = 'test'
    data = pd.concat([train, test])
    for i in range(int(total_batches)):
        data.loc[(data['time'] > i * batch) & (data['time'] <= (i + 1) * batch), 'batch'] = i + 1
    train = data[data['set'] == 'train']
    test = data[data['set'] == 'test']
    train.drop(['set'], inplace = True, axis = 1)
    test.drop(['set'], inplace = True, axis = 1)
    del data
    return train, test

def train_grouping(train, test):
  train.loc[0:1000000, 'train_group'] = 0          # batch 0 and 1
  train.loc[1000000:1500000, 'train_group'] = 1  # batch 2
  train.loc[1500000:2000000, 'train_group'] = 2  # batch 3
  train.loc[2000000:2500000, 'train_group'] = 3  # batch 4
  train.loc[2500000:3000000, 'train_group'] = 4  # batch 5
  train.loc[3000000:3500000, 'train_group'] = 1  # batch 6
  train.loc[3500000:4000000, 'train_group'] = 2  # batch 7
  train.loc[4000000:4500000, 'train_group'] = 4  # batch 8
  train.loc[4500000:5000001, 'train_group'] = 3  # batch 9

  test.loc[0:100000, 'train_group'] = 0
  test.loc[100000:200000, 'train_group'] = 2
  test.loc[200000:300000, 'train_group'] = 4
  test.loc[300000:400000, 'train_group'] = 0
  test.loc[400000:500000, 'train_group'] = 1
  test.loc[500000:600000, 'train_group'] = 3
  test.loc[600000:700000, 'train_group'] = 4
  test.loc[700000:800000, 'train_group'] = 3
  test.loc[800000:900000, 'train_group'] = 0
  test.loc[900000:1000000, 'train_group'] = 2
  test.loc[1000000:, 'train_group'] = 0

  return train, test

# def create_signal_mod(train):
#     left = 3641000
#     right = 3829000
#     thresh_dict = {
#         3: [0.1, 2.0],
#         2: [-1.1, 0.7],
#         1: [-2.3, -0.6],
#         0: [-3.8, -2],
#     }
    
#     # train['signal_mod'] = train['signal'].values
#     for ch in train[train['batch']==8]['open_channels'].unique():
#         idxs_noisy = (train['open_channels']==ch) & (left<train.index) & (train.index<right)
#         idxs_not_noisy = (train['open_channels']==ch) & ~idxs_noisy
#         mean = train[idxs_not_noisy]['signal'].mean()

#         idxs_outlier = idxs_noisy & (thresh_dict[ch][1]<train['signal'].values)
#         train['signal'][idxs_outlier]  = mean
#         idxs_outlier = idxs_noisy & (train['signal'].values<thresh_dict[ch][0])
#         train['signal'][idxs_outlier]  = mean
#     return train

# create batches of 4000 observations
def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

def train_grouping(train, test):
    train.loc[0:1000000, 'train_group'] = 0          # batch 0 and 1
    train.loc[1000000:1500000, 'train_group'] = 1  # batch 2
    train.loc[1500000:2000000, 'train_group'] = 2  # batch 3
    train.loc[2000000:2500000, 'train_group'] = 3  # batch 4
    train.loc[2500000:3000000, 'train_group'] = 4  # batch 5
    train.loc[3000000:3500000, 'train_group'] = 1  # batch 6
    train.loc[3500000:4000000, 'train_group'] = 2  # batch 7
    train.loc[4000000:4500000, 'train_group'] = 4  # batch 8
    train.loc[4500000:5000001, 'train_group'] = 3  # batch 9

    test.loc[0:100000, 'train_group'] = 0
    test.loc[100000:200000, 'train_group'] = 2
    test.loc[200000:300000, 'train_group'] = 4
    test.loc[300000:400000, 'train_group'] = 0
    test.loc[400000:500000, 'train_group'] = 1
    test.loc[500000:600000, 'train_group'] = 3
    test.loc[600000:700000, 'train_group'] = 4
    test.loc[700000:800000, 'train_group'] = 3
    test.loc[800000:900000, 'train_group'] = 0
    test.loc[900000:1000000, 'train_group'] = 2
    test.loc[1000000:, 'train_group'] = 0

    return train, test

def add_markov_metrics(train):
    # group 0
    train.loc[train['train_group'] == 0, 'from_0_to_0'] = (0.994+0.993)/2
    train.loc[train['train_group'] == 0, 'from_0_to_1'] = (0.006+0.007)/2
    train.loc[train['train_group'] == 0, 'from_1_to_0'] = (0.176+0.171)/2
    train.loc[train['train_group'] == 0, 'from_1_to_1'] = (0.824+0.829)/2

    # group 1
    train.loc[train['train_group'] == 1, 'from_0_to_0'] = (0.808+0.807)/2
    train.loc[train['train_group'] == 1, 'from_0_to_1'] = (0.192+0.193)/2
    train.loc[train['train_group'] == 1, 'from_1_to_0'] = (0.065+0.063)/2
    train.loc[train['train_group'] == 1, 'from_1_to_1'] = (0.935+0.937)/2

    # group 2
    train.loc[train['train_group'] == 2, 'from_0_to_0'] = (0.535+0.525)/2
    train.loc[train['train_group'] == 2, 'from_0_to_1'] = (0.369+0.379)/2
    train.loc[train['train_group'] == 2, 'from_0_to_2'] = (0.089+0.090)/2
    train.loc[train['train_group'] == 2, 'from_0_to_3'] = (0.006+0.007)/2
    train.loc[train['train_group'] == 2, 'from_1_to_0'] = (0.049+0.053)/2
    train.loc[train['train_group'] == 2, 'from_1_to_1'] = (0.628+0.629)/2
    train.loc[train['train_group'] == 2,'from_1_to_2'] = (0.289+0.284)/2
    train.loc[train['train_group'] == 2,'from_1_to_3'] = (0.033+0.034)/2
    train.loc[train['train_group'] == 2,'from_2_to_0'] = 0.005
    train.loc[train['train_group'] == 2,'from_2_to_1'] = (0.116+0.117)/2
    train.loc[train['train_group'] == 2,'from_2_to_2'] = (0.716+0.717)/2
    train.loc[train['train_group'] == 2,'from_2_to_3'] = 0.163
    train.loc[train['train_group'] == 2,'from_3_to_0'] = 0.
    train.loc[train['train_group'] == 2,'from_3_to_1'] = (0.016+0.017)/2
    train.loc[train['train_group'] == 2,'from_3_to_2'] = (0.191+0.194)/2
    train.loc[train['train_group'] == 2,'from_3_to_3'] = (0.792+0.788)/2

    # group 3
    train.loc[train['train_group'] == 3,'from_0_to_0'] = (0.368+0.338)/2
    train.loc[train['train_group'] == 3,'from_0_to_1'] = (0.406+0.394)/2
    train.loc[train['train_group'] == 3,'from_0_to_2'] = (0.177+0.220)/2
    train.loc[train['train_group'] == 3,'from_0_to_3'] = 0.038
    train.loc[train['train_group'] == 3,'from_0_to_4'] = (0.012+0.009)/2
    train.loc[train['train_group'] == 3,'from_0_to_5'] = 0.
    train.loc[train['train_group'] == 3,'from_1_to_0'] = (0.034+0.035)/2
    train.loc[train['train_group'] == 3,'from_1_to_1'] = (0.425+0.416)/2
    train.loc[train['train_group'] == 3,'from_1_to_2'] = (0.383+0.395)/2
    train.loc[train['train_group'] == 3,'from_1_to_3'] = (0.135+0.133)/2
    train.loc[train['train_group'] == 3,'from_1_to_4'] = (0.023+0.020)/2
    train.loc[train['train_group'] == 3,'from_1_to_5'] = 0.001
    train.loc[train['train_group'] == 3,'from_2_to_0'] = 0.003
    train.loc[train['train_group'] == 3,'from_2_to_1'] = (0.076+0.078)/2
    train.loc[train['train_group'] == 3,'from_2_to_2'] = 0.506
    train.loc[train['train_group'] == 3,'from_2_to_3'] = (0.334+0.332)/2
    train.loc[train['train_group'] == 3,'from_2_to_4'] = 0.076
    train.loc[train['train_group'] == 3,'from_2_to_5'] = 0.006
    train.loc[train['train_group'] == 3,'from_3_to_0'] = 0.
    train.loc[train['train_group'] == 3,'from_3_to_1'] = 0.011
    train.loc[train['train_group'] == 3,'from_3_to_2'] = (0.134+0.133)/2
    train.loc[train['train_group'] == 3,'from_3_to_3'] = (0.574+0.575)/2
    train.loc[train['train_group'] == 3,'from_3_to_4'] = 0.251
    train.loc[train['train_group'] == 3,'from_3_to_5'] = (0.030+0.029)/2
    train.loc[train['train_group'] == 3,'from_4_to_0'] = 0.
    train.loc[train['train_group'] == 3,'from_4_to_1'] = 0.001
    train.loc[train['train_group'] == 3,'from_4_to_2'] = (0.024+0.025)/2
    train.loc[train['train_group'] == 3,'from_4_to_3'] = (0.198+0.200)/2
    train.loc[train['train_group'] == 3,'from_4_to_4'] = (0.638+0.634)/2
    train.loc[train['train_group'] == 3,'from_4_to_5'] = 0.139
    train.loc[train['train_group'] == 3,'from_5_to_0'] = 0.
    train.loc[train['train_group'] == 3,'from_5_to_1'] = 0.
    train.loc[train['train_group'] == 3,'from_5_to_2'] = 0.004
    train.loc[train['train_group'] == 3,'from_5_to_3'] = (0.046+0.047)/2
    train.loc[train['train_group'] == 3,'from_5_to_4'] = (0.281+0.278)/2
    train.loc[train['train_group'] == 3,'from_5_to_5'] = (0.669+0.671)/2

    # group 4
    train.loc[train['train_group'] == 4,'from_0_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_1'] = 0.5
    train.loc[train['train_group'] == 4,'from_0_to_2'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_3'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_4'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_5'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_6'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_7'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_8'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_9'] = 0.
    train.loc[train['train_group'] == 4,'from_0_to_10'] = 0.
    train.loc[train['train_group'] == 4,'from_1_to_0'] = 0.035/2
    train.loc[train['train_group'] == 4,'from_1_to_1'] = (0.140+0.159)/2
    train.loc[train['train_group'] == 4,'from_1_to_2'] = (0.263+0.522)/2
    train.loc[train['train_group'] == 4,'from_1_to_3'] = (0.333+0.203)/2
    train.loc[train['train_group'] == 4,'from_1_to_4'] = (0.140+0.087)/2
    train.loc[train['train_group'] == 4,'from_1_to_5'] = (0.053+0.029)/2
    train.loc[train['train_group'] == 4,'from_1_to_6'] = 0.035/2
    train.loc[train['train_group'] == 4,'from_1_to_7'] = 0.
    train.loc[train['train_group'] == 4,'from_1_to_8'] = 0.
    train.loc[train['train_group'] == 4,'from_1_to_9'] = 0.
    train.loc[train['train_group'] == 4,'from_1_to_10'] = 0.
    train.loc[train['train_group'] == 4,'from_2_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_2_to_1'] = (0.042+0.041)/2
    train.loc[train['train_group'] == 4,'from_2_to_2'] = (0.181+0.210)/2
    train.loc[train['train_group'] == 4,'from_2_to_3'] = (0.289+0.340)/2
    train.loc[train['train_group'] == 4,'from_2_to_4'] = (0.310+0.239)/2
    train.loc[train['train_group'] == 4,'from_2_to_5'] = (0.139+0.127)/2
    train.loc[train['train_group'] == 4,'from_2_to_6'] = (0.036+0.032)/2
    train.loc[train['train_group'] == 4,'from_2_to_7'] = (0.004+0.009)/2
    train.loc[train['train_group'] == 4,'from_2_to_8'] = 0.002/2
    train.loc[train['train_group'] == 4,'from_2_to_9'] = 0.
    train.loc[train['train_group'] == 4,'from_2_to_10'] = 0.
    train.loc[train['train_group'] == 4,'from_3_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_3_to_1'] = (0.004+0.007)/2
    train.loc[train['train_group'] == 4,'from_3_to_2'] = (0.059+0.052)/2
    train.loc[train['train_group'] == 4,'from_3_to_3'] = (0.245+0.243)/2
    train.loc[train['train_group'] == 4,'from_3_to_4'] = (0.326+0.340)/2
    train.loc[train['train_group'] == 4,'from_3_to_5'] = (0.248+0.240)/2
    train.loc[train['train_group'] == 4,'from_3_to_6'] = (0.095+0.095)/2
    train.loc[train['train_group'] == 4,'from_3_to_7'] = (0.018+0.022)/2
    train.loc[train['train_group'] == 4,'from_3_to_8'] = (0.005+0.002)/2
    train.loc[train['train_group'] == 4,'from_3_to_9'] = 0.
    train.loc[train['train_group'] == 4,'from_3_to_10'] = 0.
    train.loc[train['train_group'] == 4,'from_4_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_4_to_1'] = (0.001+0.001)/2
    train.loc[train['train_group'] == 4,'from_4_to_2'] = (0.010+0.009)/2
    train.loc[train['train_group'] == 4,'from_4_to_3'] = (0.080+0.080)/2
    train.loc[train['train_group'] == 4,'from_4_to_4'] = (0.297+0.295)/2
    train.loc[train['train_group'] == 4,'from_4_to_5'] = (0.355+0.350)/2
    train.loc[train['train_group'] == 4,'from_4_to_6'] = (0.186+0.193)/2
    train.loc[train['train_group'] == 4,'from_4_to_7'] = (0.060+0.060)/2
    train.loc[train['train_group'] == 4,'from_4_to_8'] = (0.011+0.010)/2
    train.loc[train['train_group'] == 4,'from_4_to_9'] = (0.001+0.001)/2
    train.loc[train['train_group'] == 4,'from_4_to_10'] = 0.
    train.loc[train['train_group'] == 4,'from_5_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_5_to_1'] = 0.
    train.loc[train['train_group'] == 4,'from_5_to_2'] = (0.001+0.001)/2
    train.loc[train['train_group'] == 4,'from_5_to_3'] = (0.019+0.017)/2
    train.loc[train['train_group'] == 4,'from_5_to_4'] = (0.114+0.117)/2
    train.loc[train['train_group'] == 4,'from_5_to_5'] = (0.354+0.355)/2
    train.loc[train['train_group'] == 4,'from_5_to_6'] = (0.330+0.330)/2
    train.loc[train['train_group'] == 4,'from_5_to_7'] = (0.144+0.143)/2
    train.loc[train['train_group'] == 4,'from_5_to_8'] = (0.033+0.032)/2
    train.loc[train['train_group'] == 4,'from_5_to_9'] = (0.004+0.004)/2
    train.loc[train['train_group'] == 4,'from_5_to_10'] = 0.
    train.loc[train['train_group'] == 4,'from_6_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_6_to_1'] = 0.
    train.loc[train['train_group'] == 4,'from_6_to_2'] = 0.
    train.loc[train['train_group'] == 4,'from_6_to_3'] = (0.003+0.003)/2
    train.loc[train['train_group'] == 4,'from_6_to_4'] = (0.030+0.032)/2
    train.loc[train['train_group'] == 4,'from_6_to_5'] = (0.159+0.161)/2
    train.loc[train['train_group'] == 4,'from_6_to_6'] = (0.399+0.398)/2
    train.loc[train['train_group'] == 4,'from_6_to_7'] = (0.295+0.295)/2
    train.loc[train['train_group'] == 4,'from_6_to_8'] = (0.097+0.096)/2
    train.loc[train['train_group'] == 4,'from_6_to_9'] = (0.015+0.014)/2
    train.loc[train['train_group'] == 4,'from_6_to_10'] = (0.001+0.001)/2
    train.loc[train['train_group'] == 4,'from_7_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_7_to_1'] = 0.
    train.loc[train['train_group'] == 4,'from_7_to_2'] = 0.
    train.loc[train['train_group'] == 4,'from_7_to_3'] = 0.001
    train.loc[train['train_group'] == 4,'from_7_to_4'] = (0.006+0.007)/2
    train.loc[train['train_group'] == 4,'from_7_to_5'] = (0.050+0.050)/2
    train.loc[train['train_group'] == 4,'from_7_to_6'] = (0.209+0.212)/2
    train.loc[train['train_group'] == 4,'from_7_to_7'] = (0.433+0.432)/2
    train.loc[train['train_group'] == 4,'from_7_to_8'] = (0.243+0.242)/2
    train.loc[train['train_group'] == 4,'from_7_to_9'] = (0.054+0.051)/2
    train.loc[train['train_group'] == 4,'from_7_to_10'] = (0.004+0.004)/2
    train.loc[train['train_group'] == 4,'from_8_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_8_to_1'] = 0.
    train.loc[train['train_group'] == 4,'from_8_to_2'] = 0.
    train.loc[train['train_group'] == 4,'from_8_to_3'] = 0.
    train.loc[train['train_group'] == 4,'from_8_to_4'] = (0.006+0.001)/2
    train.loc[train['train_group'] == 4,'from_8_to_5'] = (0.050+0.013)/2
    train.loc[train['train_group'] == 4,'from_8_to_6'] = (0.209+0.074)/2
    train.loc[train['train_group'] == 4,'from_8_to_7'] = (0.433+0.264)/2
    train.loc[train['train_group'] == 4,'from_8_to_8'] = (0.243+0.456)/2
    train.loc[train['train_group'] == 4,'from_8_to_9'] = (0.054+0.173)/2
    train.loc[train['train_group'] == 4,'from_8_to_10'] = (0.004+0.019)/2
    train.loc[train['train_group'] == 4,'from_9_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_9_to_1'] = 0.
    train.loc[train['train_group'] == 4,'from_9_to_2'] = 0.
    train.loc[train['train_group'] == 4,'from_9_to_3'] = 0.
    train.loc[train['train_group'] == 4,'from_9_to_4'] = 0.
    train.loc[train['train_group'] == 4,'from_9_to_5'] = (0.003+0.002)/2
    train.loc[train['train_group'] == 4,'from_9_to_6'] = (0.019+0.021)/2
    train.loc[train['train_group'] == 4,'from_9_to_7'] = (0.102+0.103)/2
    train.loc[train['train_group'] == 4,'from_9_to_8'] = (0.310+0.317)/2
    train.loc[train['train_group'] == 4,'from_9_to_9'] = (0.471+0.461)/2
    train.loc[train['train_group'] == 4,'from_9_to_10'] = (0.094+0.095)/2
    train.loc[train['train_group'] == 4,'from_10_to_0'] = 0.
    train.loc[train['train_group'] == 4,'from_10_to_1'] = 0.
    train.loc[train['train_group'] == 4,'from_10_to_2'] = 0.
    train.loc[train['train_group'] == 4,'from_10_to_3'] = 0.
    train.loc[train['train_group'] == 4,'from_10_to_4'] = 0.
    train.loc[train['train_group'] == 4,'from_10_to_5'] = 0.001
    train.loc[train['train_group'] == 4,'from_10_to_6'] = (0.003+0.005)/2
    train.loc[train['train_group'] == 4,'from_10_to_7'] = (0.028+0.031)/2
    train.loc[train['train_group'] == 4,'from_10_to_8'] = (0.134+0.132)/2
    train.loc[train['train_group'] == 4,'from_10_to_9'] = (0.357+0.361)/2
    train.loc[train['train_group'] == 4,'from_10_to_10'] = (0.478+0.470)/2

    return train

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    # train['signal'] = train.signal / 15.0
    # test['signal'] = test.signal / 15.0
    
    return train, test

def add_preprocessing_data(train, test):
    pre_train = pd.read_pickle("./data/pre_train.pkl")
    pre_test = pd.read_pickle("./data/pre_test.pkl")
    train = pd.concat([train, pre_train], axis=1)
    test = pd.concat([test, pre_test], axis=1)
    
    del pre_train, pre_test
    gc.collect()
    
    return train, test

def add_category(train, test):
  train["category"] = 0
  test["category"] = 0

  # train segments with more then 9 open channels classes
  train.loc[2000000:2500000-1, 'category'] = 1
  train.loc[4500000:5000000-1, 'category'] = 1

  # test segments with more then 9 open channels classes (potentially)
  test.loc[500000:600000-1, "category"] = 1
  test.loc[700000:800000-1, "category"] = 1
  
  return train, test

# signal processing features
def calc_gradients(s, n_grads = 4, is_clean=True):
    '''
    Calculate gradients for a pandas series. Returns the same number of samples
    '''
    grads = pd.DataFrame()

    col='grad_'
    if is_clean:
      col = 'clean_grad_'
    
    g = s.values
    for i in range(n_grads):
        g = np.gradient(g)
        grads[col + str(i+1)] = g
        
    return grads

def calc_low_pass(s, n_filts=10, is_clean=True):
    '''
    Applies low pass filters to the signal. Left delayed and no delayed
    '''
    wns = np.logspace(-2, -0.3, n_filts)

    col='lowpass_'
    if is_clean:
      col = 'clean_lowpass_'
    
    low_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='low')
        zi = signal.lfilter_zi(b, a)
        low_pass[col + 'lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        low_pass[col + 'ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return low_pass

def calc_high_pass(s, n_filts=10, is_clean=True):
    '''
    Applies high pass filters to the signal. Left delayed and no delayed
    '''
    wns = np.logspace(-2, -0.1, n_filts)

    col='highpass_'
    if is_clean:
      col = 'clean_highpass_'
    
    high_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='high')
        zi = signal.lfilter_zi(b, a)
        high_pass[col + 'lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        high_pass[col + 'ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return high_pass

def calc_ewm(s, windows=[10, 50, 100, 500, 1000], is_clean=True):
    '''
    Calculates exponential weighted functions
    '''
    ewm = pd.DataFrame()

    col = 'ewm_'
    if is_clean:
      col = 'clean_ewm_'

    for w in windows:
        ewm[col + 'mean_' + str(w)] = s.ewm(span=w, min_periods=1).mean()
        ewm[col + 'std_' + str(w)] = s.ewm(span=w, min_periods=1).std()
        
    # add zeros when na values (std)
    ewm = ewm.fillna(value=0)
        
    return ewm


def add_features(s, is_clean=True):
    '''
    All calculations together
    '''
    if is_clean:
      low_pass = calc_low_pass(s, is_clean=True)
      high_pass = calc_high_pass(s, is_clean=True)
      # gradients = calc_gradients(s, is_clean=True)
      # ewm = calc_ewm(s, is_clean=True)
      return pd.concat([s, low_pass, high_pass], axis=1)
    else:
      gradients = calc_gradients(s, is_clean=False)
      ewm = calc_ewm(s, is_clean=False)
      return pd.concat([s, gradients, ewm], axis=1)

def divide_and_add_features(s, signal_size=500000, is_clean=True):
    '''
    Divide the signal in bags of "signal_size".
    Normalize the data dividing it by 15.0
    '''
    # normalize
    # s = s / 15.0
    
    ls = []
    for i in tqdm(range(int(s.shape[0]/signal_size))):
        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)
        sig_featured = add_features(sig, is_clean=is_clean)
        ls.append(sig_featured)
    
    return pd.concat(ls, axis=0)

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

def calc_roll_stats(df, windows, group='group'):
    '''
    Calculates rolling stats like mean, std, min, max...
    '''
    for i, window in enumerate(windows):
      df[group + 'roll_mean_' + str(window)] = df.groupby(group)['signal'].rolling(window=window, min_periods=1).mean().fillna(0).values
      df[group + 'roll_std_' + str(window)] = df.groupby(group)['signal'].rolling(window=window, min_periods=1).std().fillna(0).values
      df[group + 'roll_min_' + str(window)] = df.groupby(group)['signal'].rolling(window=window, min_periods=1).min().fillna(0).values
      df[group + 'roll_max_' + str(window)] = df.groupby(group)['signal'].rolling(window=window, min_periods=1).max().fillna(0).values
      df[group + 'roll_range' + str(window)] = df[group + 'roll_max_' + str(window)] - df[group + 'roll_min_' + str(window)]

      df['roll_q10_' + str(window)] = df.groupby('group')['signal'].rolling(window=window, min_periods=1).quantile(0.10).fillna(0).values
      df['roll_q25_' + str(window)] = df.groupby('group')['signal'].rolling(window=window, min_periods=1).quantile(0.25).fillna(0).values
      df['roll_q50_' + str(window)] = df.groupby('group')['signal'].rolling(window=window, min_periods=1).quantile(0.50).fillna(0).values
      df['roll_q75_' + str(window)] = df.groupby('group')['signal'].rolling(window=window, min_periods=1).quantile(0.75).fillna(0).values
      df['roll_q90_' + str(window)] = df.groupby('group')['signal'].rolling(window=window, min_periods=1).quantile(0.90).fillna(0).values
             
    return df

def calc_expand_stats(df, group='group'):
  df['expanding_mean'] = df.groupby(group)['signal'].expanding().mean().fillna(0).values
  df['expanding_std'] = df.groupby(group)['signal'].expanding().std().fillna(0).values
  df['expanding_max'] = df.groupby(group)['signal'].expanding().max().fillna(0).values
  df['expanding_min'] = df.groupby(group)['signal'].expanding().min().fillna(0).values
  df['expanding_range'] = df['expanding_max'] - df['expanding_min']
  
  return df

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)

    # create leads and lags
    df = lag_with_pct_change(df, np.asarray(range(1, 3), dtype=np.int32))

    # create rolling stats
    df = calc_roll_stats(df, [100]) # groupごと(4000)
    # df = calc_roll_stats(df, [50000, 100000], group='batch') # batchごと(500000)

    # create expanding stats
    # df = calc_expand_stats(df)

    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2

    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time', 'batch', 'train_group', 'test_group']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features

# model function (very important, you can try different arquitectures to get a better score. I believe that top public leaderboard is a 1D Conv + RNN style)
def Classifier(shape_):
    
    def cbr(x, out_layer, kernel, stride, dilation):
        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
    
    inp = Input(shape = (shape_))
    x = cbr(inp, 64, 7, 1, 1)
    x = BatchNormalization()(x)
    x = wave_block(x, 16, 3, 12)
    x = BatchNormalization()(x)
    x = wave_block(x, 32, 3, 8)
    x = BatchNormalization()(x)
    x = wave_block(x, 64, 3, 4)
    x = BatchNormalization()(x)
    x = wave_block(x, 128, 3, 1)
    x = cbr(x, 32, 7, 1, 1)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = LR)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
    return model

# function that decrease the learning as epochs increase (i also change this part of the code)
def lr_schedule(epoch):
    if epoch < 30:
        lr = LR
    elif epoch < 40:
        lr = LR / 3
    elif epoch < 50:
        lr = LR / 5
    elif epoch < 60:
        lr = LR / 7
    elif epoch < 70:
        lr = LR / 9
    elif epoch < 80:
        lr = LR / 11
    elif epoch < 90:
        lr = LR / 13
    else:
        lr = LR / 100
    return lr

# class to get macro f1 score. This is not entirely necessary but it's fun to check f1 score of each epoch (be carefull, if you use this function early stopping callback will not work)
class MacroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis = 2).reshape(-1)
        
    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)
        score = f1_score(self.targets, pred, average = 'macro')
        print(f'F1 Macro Score: {score:.5f}')

def compensateTarget(df):
  for i in range(11):
    if i not in df.columns:
      df[i] = 0

  df = df.loc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'group']]

  return df

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = int(np.max(y) + 1)
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[int(g)][int(label)] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.
def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    seed_everything(SEED)
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)
    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
    group = train['group']

    # kf = GroupKFold(n_splits=5)
    # splits = [x for x in kf.split(train, train[target], group)]
    # kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    # splits = [x for x in kf.split(train, train[target])]
    splits = stratified_group_k_fold(train, train.open_channels, group, k=5, seed=42)

    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

    tr = compensateTarget(tr)

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))

    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')

        gc.collect()
        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)
        model = Classifier(shape_)
        # using our lr_schedule function
        cb_lr_schedule = LearningRateScheduler(lr_schedule)
        model.fit(train_x,train_y,
                  epochs = nn_epochs,
                  callbacks = [cb_lr_schedule, MacroF1(model, valid_x, valid_y)], # adding custom evaluation metric for each epoch
                  batch_size = nn_batch_size,verbose = 2,
                  validation_data = (valid_x,valid_y))
        preds_f = model.predict(valid_x)
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability
        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx,:] += preds_f
        te_preds = model.predict(test)
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / SPLITS

    # return oof_, preds_

    # calculate the oof macro f1_score
    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')
    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    sample_submission.to_csv('submission_wavenet.csv', index=False, float_format='%.4f')

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        if col!='open_channels':
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

from sklearn.linear_model import LinearRegression

def predictGeneralMean(signal_, channels, c):
  label = np.arange(len(signal_))

  channel_list = np.arange(c)
  n_list = np.empty(c)
  mean_list = np.empty(c)
  std_list = np.empty(c)
  stderr_list = np.empty(c)

  for i in range(c):
      x = label[channels == i]
      y = signal_[channels == i]
      n_list[i] = np.size(y)
      mean_list[i] = np.mean(y)
      std_list[i] = np.std(y)
      
  stderr_list = std_list / np.sqrt(n_list)

  w = 1 / stderr_list
  channel_list = channel_list.reshape(-1, 1)
  linreg_m = LinearRegression()
  linreg_m.fit(channel_list, mean_list, sample_weight = w)

  mean_predict = linreg_m.predict(channel_list)

  # print("mean :", mean_predict)

  return mean_predict

def Arrange_mean(signal_, channels, diff, channel_range):
    signal_out = signal_.copy()
    for i in range(channel_range):
        signal_out[channels == i] -= diff[i]
    return signal_out

def bandstop(x, samplerate = 1000000, fp = np.array([4925, 5075]), fs = np.array([4800, 5200])):
    fn = samplerate / 2   # Nyquist frequency
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 10.0

    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "bandstop")
    y = signal.filtfilt(b, a, x)
    return y

def remove_noise_by_bandstop(df):
  signalA = df[df['category']==0].signal.values
  signalB = df[df['category']==1].signal.values

  channelsA = df[df['category']==0].open_channels.values
  channelsB = df[df['category']==1].open_channels.values

  c_A = np.unique(channelsA).size # channelの種類
  c_B = np.unique(channelsB).size # channelの種類

  mean_predictA = predictGeneralMean(signalA, channelsA, c_A)
  mean_predictB = predictGeneralMean(signalB, channelsB, c_B)

  sig_A = Arrange_mean(signalA, channelsA, mean_predictA, c_A)
  sig_B = Arrange_mean(signalB, channelsB, mean_predictB, c_B)

  signal_flat = np.hstack((sig_A, sig_B))
  # sig_list = np.split(signal_flat, 9)
  sig_list = np.split(signal_flat, 10)

  for i, sig_sample in enumerate(sig_list):
      # batch 1, 2, 3, 4, 6, 7, 8, 9, 5, 10 の順番
      sig_remove = bandstop(sig_sample)
      df[df['batch']==i+1]['signal'] = sig_remove

  return df

print('Reading Data Started...')
train, test, sample_submission = read_data()

train, test = train_grouping(train, test)
train = add_markov_metrics(train)
test = add_markov_metrics(test)
train.fillna(0)
test.fillna(0)

train, test = add_category(train, test)
train, test = batching_10(train, test)
#train = create_signal_mod(train)
train = remove_noise_by_bandstop(train)

train, test = normalize(train, test)

print('Reduce memory usage...')
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

print('Reading and Normalizing Data Completed')

# train. test = train_grouping(train, test)
# trn, tst = target_encode(train["train_group"], test["train_group"], target=train.open_channels, min_samples_leaf=100, smoothing=10, noise_level=0.01)
# train['target_encoding_group'] = trn.values
# test['target_encoding_group'] = tst.values

print('Creating Features')
print('Feature Engineering Started...')

train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)
test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)

# pre_train = divide_and_add_features(train.signal, signal_size=500000, is_clean=False)
# pre_test = divide_and_add_features(test.signal, signal_size=500000, is_clean=False)
# pre_train.drop('signal', axis=1, inplace=True)
# pre_test.drop('signal', axis=1, inplace=True)
# pre_train.reset_index(inplace = True, drop = True)
# pre_test.reset_index(inplace = True, drop = True)
# train = pd.concat([train, pre_train], axis=1)
# test = pd.concat([test, pre_test], axis=1)

train, test = add_preprocessing_data(train, test)

# tr_clean = pd.read_csv('./data/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
# ts_clean = pd.read_csv('./data/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
# tr_clean, ts_clean = batching_10(tr_clean, ts_clean)
# tr_clean, ts_clean = add_category(tr_clean, ts_clean)
# # tr_clean = create_signal_mod(tr_clean)
# tr_clean = remove_noise_by_bandstop(tr_clean)
# tr_clean, ts_clean = normalize(tr_clean, ts_clean)
# pre_train = divide_and_add_features(tr_clean.signal, signal_size=500000, is_clean=True)
# pre_test = divide_and_add_features(ts_clean.signal, signal_size=500000, is_clean=True)
# pre_train.drop('signal', axis=1, inplace=True)
# pre_test.drop('signal', axis=1, inplace=True)
# pre_train.reset_index(inplace = True, drop = True)
# pre_test.reset_index(inplace = True, drop = True)
# train = pd.concat([train, pre_train], axis=1)
# test = pd.concat([test, pre_test], axis=1)

# del pre_train, pre_test, tr_clean, ts_clean
# gc.collect

train, test, features = feature_selection(train, test)
print('Feature Engineering Completed...')

print('Reduce memory usage...')
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

# trn, tst = target_encode(train["category"], test["category"], target=train.open_channels, min_samples_leaf=100, smoothing=10, noise_level=0.01)
# train['target_encoding'] = trn.values
# test['target_encoding'] = tst.values

print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')
run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)
print('Training completed...')
