import torch
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import roc_auc_score
import numpy as np
from time import time, sleep

from pyod.models.iforest import IForest
from tqdm import tqdm

from sklearn.preprocessing import QuantileTransformer
from pfns.utils import normalize_data

from config import *
from fomox import load_model, create_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from eval_data import iter_adbench

import os

import json

def _predict(model,x,test):
    x=x.transpose(1,0)
    y=None
    with torch.no_grad():
        logits=model(x,y,test.transpose(1,0), only_return_standard_out=False)
    logits={k:v.cpu().numpy()[:,0] for k,v in logits.items()}
    return logits

def predict(train, test, device=device, model="fomox.ckpt"):
    if type(model) is str:
        model=load_model(ckpt).to(device)
    train=torch.tensor(train).float().to(device).unsqueeze(0)
    test=torch.tensor(test).float().to(device).unsqueeze(0)
    return _predict(model, train, test)
