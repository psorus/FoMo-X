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

def predict(model,x,test):
    x=x.transpose(1,0)
    y=None
    with torch.no_grad():
        logits=model(x,y,test.transpose(1,0), only_return_standard_out=False)
    logits={k:v.cpu().numpy()[:,0] for k,v in logits.items()}
    return logits

#on gpu, loading the adbench data takes most of the time. We cache it.
import pickle
if os.path.exists("quickbench.pkl"):
    with open("quickbench.pkl","rb") as f:
        data=pickle.load(f)
else:
    data=[zw for zw in tqdm(iter_adbench(),desc="Loading ADBENCH data")]
    with open("quickbench.pkl","wb") as f:
        pickle.dump(data,f)

save={}
def eval_model(model, quiet=True):
    times=0.0
    samples=0
    
    aucs=[]
    
    ut0=time()
    for fn, (x,tx,ty) in data:
        x=torch.tensor(x).float().to(device).unsqueeze(0)
        tx=torch.tensor(tx).float().to(device).unsqueeze(0)
        t0=time()
        dic=predict(model, x, tx)
        t1=time()
        pred=dic["standard"]
        for key in dic:
            save[f"{fn}_{key}"]=dic[key]
        times+=t1-t0
        samples+=len(ty)
        auc=roc_auc_score(ty, pred[:,1])
        aucs.append(auc)
        save[f"{fn}_auc"]=auc
        save[f"{fn}_gt"]=ty
        if not quiet:print("dataset",fn, "reached AUC:", auc)
    
    script_time=time()-ut0
    

    return {"avg_auc": float(np.mean(aucs)),
            "time_per_sample_ms": 1000*times/samples,
            "time_per_sample_mus": 1000*1000*times/samples,
            "total_time_s": times,
            "script_time_s": script_time
            }

if __name__ == "__main__":
    model=load_model()
    #del model.decoder_dict["severity"]
    #del model.decoder_dict["variation"]
    model.to(device)
    model.eval()
    print(json.dumps(eval_model(model, quiet=False),indent=2))
    np.savez_compressed("results.npz", **save)
