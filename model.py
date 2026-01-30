import torch
from torch import nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pfns import TransformerModel, encoders, positional_encodings

from sklearn.metrics import roc_auc_score
import numpy as np
from time import time

from pyod.models.iforest import IForest
from tqdm import tqdm



from config import *

def create_model():

    encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear)
    y_encoder_generator=encoders.Linear
    pos_encoder=positional_encodings.NoPositionalEncoding(emsize, seq_len*2)
    
    criterion = nn.CrossEntropyLoss(weight=torch.ones(size=(num_class,)) / num_class, reduction='none',
                                        ignore_index=-33)#hps['ignore_index'])
    
    encoder = encoder_generator(num_features, emsize)
    model = TransformerModel(encoder=encoder
                                  #, criterion=criterion
                                  , nhead=nhead
                                  , ninp=emsize
                                  , nhid=nhid
                                  , nlayers=nlayers
                                  , dropout=dropout
                                  , style_encoder=style_encoder
                                  , y_encoder=y_encoder_generator(1, emsize)
                                  , input_normalization=input_normalization
                                  , pos_encoder=pos_encoder
                                  , decoder_dict=decoder_dict
                                  , init_method=initializer
                                  , efficient_eval_masking=efficient_eval_masking
                                  , decoder_once_dict=decoder_once_dict
                                  , num_global_att_tokens=num_global_att_tokens
                                  , **model_extra_args
                                  )
    model.criterion=criterion
    return model

def load_model(ckpt="original_fomo.ckpt", model=None):
    model=create_model() if model is None else model
        
    
    sM = model.state_dict()
    sL = torch.load(ckpt, map_location=device)
    if "state_dict" in sL:sL=sL["state_dict"]
    
    toload={}
    
    fails=0
    works=0
    for key in sorted(sL.keys()):
        key2=key.replace("model.","")
        toload[key2]=sL[key]
        if key2 in sM:
            if sM[key2].shape==sL[key].shape:
                works+=1
                continue
        print(key, sL[key].shape, "->", sM[key2].shape if key2 in sM else "MISSING")
        fails+=1
    
    #print("Fails:", fails, "Works:", works)
    #print("lenM", len(sM), "lenL", len(sL))
    
    #print(sM.keys())
    
    
    
    model.load_state_dict(toload, strict=True)  # strict=False allows missing keys

    return model
    
    
    
    
    
