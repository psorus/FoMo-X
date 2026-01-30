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
    
    sev_head=SeverityHead(emsize, num_classes=4)
    var_head=VariationHead(emsize, num_classes=1)
    model.decoder_dict['severity']=sev_head
    model.decoder_dict['variation']=var_head


    return model

def load_model(ckpt="fomox.ckpt", model=None):
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
    
    
    model.load_state_dict(toload, strict=True)  

    return model
    
    
    
    
    
