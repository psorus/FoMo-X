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
from model import load_model, create_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os

import json


class VariationHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=nhid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
class SeverityHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=nhid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)
def save_model(model, name):
    torch.save(model.state_dict(), name)
def load_variation():
    model=create_model()
    new_head= VariationHead(
        input_dim=model.ninp,
        num_classes=1,
    )
    new_head.to(device)

    model.decoder_dict["variation"]=new_head


    pth="models/variation/USE/best_corr.pth"
    model=load_model(pth, model=model)
    model.to(device)
    return model

def load_severity():
    model=create_model()
    new_head= SeverityHead(
        input_dim=model.ninp,
        num_classes=4,
    )
    new_head.to(device)

    model.decoder_dict["severity"]=new_head


    pth="models/severity/USE/best_bacc.pth"
    model=load_model(pth, model=model)
    model.to(device)
    return model

if __name__ == "__main__":

    variation_model=load_variation()
    severity_model=load_severity()

    assemble_model=load_model("fomo.ckpt")


    assemble_model.decoder_dict["variation"]=variation_model.decoder_dict["variation"]
    assemble_model.decoder_dict["severity"]=severity_model.decoder_dict["severity"]

    assemble_model.to(device)

    save_model(assemble_model, "models/fomox.ckpt")

