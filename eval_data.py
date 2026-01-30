import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


import numpy as np


from sklearn.preprocessing import QuantileTransformer
from pfns.utils import normalize_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from glob import glob
import os

from config import *

import warnings


#For this to work, please clone ADBench into the current folder
adatas=glob("ADBench/adbench/datasets/Classical/*npz")+glob("ADBench/adbench/datasets/CV_by_ResNet18/*.npz")+glob("ADBench/adbench/datasets/NLP_by_BERT/*.npz")


#Data preprocessing steps
def feature_scale(x, num_feature, rescale_with_sqrt=False):
    if rescale_with_sqrt:
        return x / (num_feature / max_feature_dim) ** (1 / 2)
    return x / (num_feature / max_feature_dim)
def feature_subsampling(x, num_feature):
    # inliners and anomalies will share the same sub-samping order, so we expect x to be the concatenation of both
    # during inference on real dataset
    x = x[:, sorted(np.random.choice(num_feature, max_feature_dim, replace=False))]
    # from line 366 of tabpfn_interface.py
    # eval_xs = eval_xs[:, :, sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False))]
    return x
def pfn_inference_transform(eval_xs: np.ndarray, preprocess_transform: str, eval_position: int,
                            normalize_with_test: bool = False, rescale_with_sqrt: bool = False):
    if len(eval_xs.shape) != 2:
        raise Exception(
            "Transforms only allow input of shape: (#sampled, #feat), but we have: {}".format(eval_xs.shape))

    num_feature = eval_xs.shape[-1]
    if num_feature > max_feature_dim:
        eval_xs = feature_subsampling(x=eval_xs, num_feature=num_feature)

    pt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, eval_position), random_state=0)

    eval_xs = torch.from_numpy(eval_xs)  # convert to torch
    eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)
    # perform (x-mean)/std normalization
    eval_xs = eval_xs.cpu().numpy()  # convert back to numpy

    warnings.simplefilter('error')
    if preprocess_transform != 'none':
        #print('feature preprocessing transform with {}'.format(preprocess_transform))
        pt.fit(eval_xs[0:eval_position,:])
        eval_xs=pt.transform(eval_xs)
    warnings.simplefilter('default')

    eval_xs = feature_scale(x=eval_xs, num_feature=eval_xs.shape[-1], rescale_with_sqrt=rescale_with_sqrt)
    if eval_xs.shape[-1] < max_feature_dim:
        eval_xs = np.concatenate([eval_xs, np.zeros(shape=(eval_xs.shape[0], max_feature_dim - num_feature))],
                                 axis=-1)
    return eval_xs
def make_train_test(x_train, y_train, x_test, y_test):
    if x_train.shape[0] <= seq_len - 1:
        train_x = x_train
    else:
        train_sub_indices = np.random.choice(x_train.shape[0], seq_len - 1, replace=False)
        train_x = x_train[train_sub_indices]

    train_and_test = pfn_inference_transform(eval_xs=np.concatenate([train_x, x_test], axis=0),
                                            preprocess_transform=preprocess_transform,
                                            eval_position=len(train_x),
                                            normalize_with_test=False,
                                            rescale_with_sqrt=False)

    train_x, x_test = train_and_test[:len(train_x), :], train_and_test[len(train_x):, :]
    return train_x, x_test, y_test
def handle_XY(X,y):
    normal=X[y==0]
    if len(normal)>50000:
        normal=normal[np.random.choice(len(normal), 50000, replace=False)]
    abnormal=X[y==1]
    normal_train, normal_test=train_test_split(normal, test_size=0.5, random_state=0)
    
    X_train=normal_train
    y_train=np.zeros(X_train.shape[0], dtype=int)
    X_test=np.concatenate([normal_test, abnormal], axis=0)
    y_test=np.concatenate([np.zeros(normal_test.shape[0], dtype=int), np.ones(abnormal.shape[0], dtype=int)], axis=0)
    
    
    scaler=StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    return  make_train_test(X_train, y_train, X_test, y_test)

def handle_xtx(x, tx, ty):
    y=np.zeros(len(x), dtype=int)

    scaler=StandardScaler().fit(x)
    x=scaler.transform(x)
    tx=scaler.transform(tx)

    return make_train_test(x, y, tx, ty)

def handle_one(datapth):
    f=np.load(datapth)
    X,y=f["X"],f["y"]
    return handle_XY(X,y)

def iter_adbench():
    for fn in adatas:
        yield os.path.basename(fn),handle_one(fn)

