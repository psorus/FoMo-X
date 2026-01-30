#idea: calculate sigma only after the softmax layer
import torch
from torch import nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import *
from model import create_model, load_model
from generate_data_variation import generate_batch, generate_dataset, lin_tuple

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import wandb
import math

from time import time
import os

from scipy.stats import spearmanr

learning_rate=0.001
num_epochs=200
batch_size=8
batch_count=250
validation_size=batch_count*batch_size//4
save_path="models/variation/USE/"
predicted_classes=1
head_name="variation"

data_hyper={
    "feature_dim":[5,max_feature_dim],
    "delta":[-0.1,0.1],
    "max_mean":5.0,
    "min_variance_scale":0.25,
    "max_variance_scale":2.0,
    "min_variance_uniformity":0.1,
    "max_variance_uniformity":1.0,
    }


os.makedirs(save_path, exist_ok=True)



wandb.init(
    project="fomox-variation",
    name="USE",
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "validation_size": validation_size,
        "save_path": save_path,
        "data_hyper": data_hyper,
    }
)


from torch.optim.lr_scheduler import StepLR



model=load_model("fomo.ckpt").to(device)

for param in model.parameters():
    param.requires_grad = False

def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return e_x / e_x.sum(dim=-1, keepdim=True)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=nhid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

new_head= ClassificationHead(
    input_dim=model.ninp,
    num_classes=predicted_classes,
)
new_head.to(device)

model.decoder_dict[head_name]=new_head

model.train()### 

model.decoder_dict[head_name].train()


optimizer = torch.optim.Adam(model.decoder_dict[head_name].parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

criterion = nn.L1Loss()
wandb.watch(model, log="all", log_freq=500)

def prep_data(X,Xa,test_size=0.5):
    indices=torch.randperm(X.size(1))
    test_size=float(lin_tuple(test_size))
    train_ind, test_ind=train_test_split(indices, test_size=test_size, random_state=42)
    train_X, test_X= X[:,train_ind], X[:,test_ind]

    train=train_X
    train_Y=None
    test=torch.cat([test_X, Xa], dim=1)
    test_Y=torch.cat([torch.zeros((test_X.size(0),test_X.size(1)), dtype=Xa.dtype), torch.ones((Xa.size(0),Xa.size(1)), dtype=Xa.dtype)], dim=1).to(device)
    test_Y=test_Y.long()
    return train, train_Y, test, test_Y

def save_model(model, name):
    torch.save(model.state_dict(), name)

def apply_model(model, train, train_Y, test):
    with torch.no_grad():
        model.train()
        sigma=[]
        for i in range(10):
            outputs = model(train.transpose(1,0), train_Y, test.transpose(1,0),
                                 only_return_standard_out=True)
            sigma.append(softmax(outputs.detach()))
        model.eval()
    outputs = model(train.transpose(1,0), train_Y, test.transpose(1,0),
                         only_return_standard_out=False)
    
    outputs=outputs[head_name]
    outputs=outputs.transpose(1,0)
    sigma=torch.stack(sigma, dim=0)
    sigma=torch.std(sigma, dim=0).transpose(1,0)
    sigma=sigma[:,:,0]
    sigma=torch.log(torch.abs(sigma)+1e-6)

    return outputs, sigma


#create validation dataset
ValidationData=generate_batch(count=validation_size, samples=seq_len, **data_hyper)
ValidationData=[zw.to(device) for zw in ValidationData]
val_train, val_train_Y, val_test, val_test_Y =prep_data(*ValidationData, test_size=0.333)

best_train_loss=float('inf')
best_val_loss=float('inf')
best_val_corr=-2.0
for epoch in range(num_epochs):

    t0=time()
    TrainData=generate_batch(count=batch_size*batch_count, samples=seq_len, **data_hyper)
    TrainData=[zw.to(device) for zw in TrainData]
    data_time=time()-t0
   
    t0=time()
    losses=[]
    for batch in tqdm(range(batch_count), desc=f"Epoch {epoch+1}/{num_epochs}"):
        BatchData=[zw[batch*batch_size:(batch+1)*batch_size] for zw in TrainData]
    
        train, train_Y, test, test_Y =prep_data(*BatchData, test_size=0.5)
    
        optimizer.zero_grad()

        predictions, goal = apply_model(model, train, train_Y, test)


        prediction_view = predictions.reshape(-1, predicted_classes)
        targets = goal.reshape(-1, predicted_classes)
        loss = criterion(prediction_view, targets)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    train_time=time()-t0

    t0=time()
    # --- VALIDATION LOOP ---
    model.eval()
    with torch.no_grad():
        val_loss = []
        #val_corr = []

        y_pred=[]
        y_true=[]

        
        for i in tqdm(range(0,validation_size, batch_size), desc="Validating"):
            val_predictions, val_goal=apply_model(model, val_train[i:i+batch_size], val_train_Y, val_test[i:i+batch_size])
            
            val_prediction_view = val_predictions.reshape(-1, predicted_classes)
            val_target_view = val_goal.reshape(-1, predicted_classes)

            loss_val = criterion(val_prediction_view, val_target_view)
            val_loss.append(loss_val.item())

            y_pred.append(val_prediction_view.cpu().numpy())
            y_true.append(val_target_view.cpu().numpy())


        y_pred=np.concatenate(y_pred, axis=0)
        y_true=np.concatenate(y_true, axis=0)

        val_corr=spearmanr(y_pred.flatten(), y_true.flatten()).correlation
            
        val_loss=np.mean(val_loss)

    
    model.decoder_dict[head_name].train()
    val_time=time()-t0

    train_loss=np.mean(losses)

    total_time=data_time+train_time+val_time

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Corr: {val_corr:.4f}')
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_corr": val_corr,
        "z_data_time": data_time,
        "z_train_time": train_time,
        "z_val_time": val_time,
        "total_time": total_time,
        "z_lr": scheduler.get_last_lr()[0],
    })

    save_model(model, os.path.join(save_path, "last.pth"))
    if val_loss < best_val_loss:
        best_val_loss=val_loss
        save_model(model, os.path.join(save_path, "best_loss.pth"))
    if train_loss < best_train_loss:
        best_train_loss=train_loss
        save_model(model, os.path.join(save_path, "best_train.pth"))
    if val_corr > best_val_corr:
        best_val_corr=val_corr
        save_model(model, os.path.join(save_path, "best_corr.pth"))






wandb.finish()





