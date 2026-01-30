import torch
from torch import nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import *
from model import create_model, load_model
from generate_data_severity import generate_batch, generate_dataset, lin_tuple

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
validation_size=batch_count*batch_size
save_path="models/severity/USE/"
predicted_classes=4
head_name="severity"

data_hyper={
    "feature_dim":[5,max_feature_dim],
    "delta":0,
    "max_mean":5.0,
    "min_variance_scale":0.25,
    "max_variance_scale":2.0,
    "min_variance_uniformity":0.1,
    "max_variance_uniformity":1.0,
    }


os.makedirs(save_path, exist_ok=True)



wandb.init(
    project="fomox-severity",
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

criterion = nn.CrossEntropyLoss()
wandb.watch(model, log="all", log_freq=500)

def prep_data(X,Xa,sevX, sevXa, test_size=0.5):
    indices=torch.randperm(X.size(1))
    test_size=float(lin_tuple(test_size))
    train_ind, test_ind=train_test_split(indices, test_size=test_size, random_state=42)
    train_X, test_X= X[:,train_ind], X[:,test_ind]
    sev_train_X, sev_test_X= sevX[:,train_ind], sevX[:,test_ind]

    train=train_X
    sev_train=sev_train_X
    train_Y=None
    test=torch.cat([test_X, Xa], dim=1)
    sev_test=torch.cat([sev_test_X, sevXa], dim=1)
    test_Y=torch.cat([torch.zeros((test_X.size(0),test_X.size(1)), dtype=Xa.dtype), torch.ones((Xa.size(0),Xa.size(1)), dtype=Xa.dtype)], dim=1).to(device)
    test_Y=test_Y.long()

    return train, train_Y, test, test_Y, sev_test

def save_model(model, name):
    torch.save(model.state_dict(), name)

def apply_model(model, train, train_Y, test):
    outputs = model(train.transpose(1,0), train_Y, test.transpose(1,0),
                         only_return_standard_out=False)
    outputs=outputs[head_name]
    outputs=outputs.transpose(1,0)


    return outputs


#create validation dataset
ValidationData=generate_batch(count=validation_size, samples=seq_len, device="cpu", **data_hyper)
ValidationData=[zw.to(device) for zw in ValidationData]
val_train, val_train_Y, val_test, val_test_Y, val_sev_test=prep_data(*ValidationData, test_size=0.5)

best_train_loss=float('inf')
best_val_loss=float('inf')
best_val_acc=float('-inf')
best_val_bacc=float('-inf')
for epoch in range(num_epochs):

    t0=time()
    TrainData=generate_batch(count=batch_size*batch_count, samples=seq_len, device="cpu",**data_hyper)
    TrainData=[zw.to(device) for zw in TrainData]
    data_time=time()-t0
   
    t0=time()
    losses=[]
    for batch in tqdm(range(batch_count), desc=f"Epoch {epoch+1}/{num_epochs}"):
        BatchData=[zw[batch*batch_size:(batch+1)*batch_size] for zw in TrainData]
    
        train, train_Y, test, test_Y,sev_test=prep_data(*BatchData, test_size=0.5)
    
        optimizer.zero_grad()

        predictions = apply_model(model, train, train_Y, test)


        prediction_view = predictions.reshape(-1, predicted_classes)
        targets = torch.argmax(sev_test, dim=-1).reshape(-1)
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
        val_acc = []
        val_sub_right=[0.0 for _ in range(predicted_classes)]
        val_sub_total=[0.0 for _ in range(predicted_classes)]
        #val_corr = []

        y_pred=[]
        y_true=[]

        
        for i in tqdm(range(0,validation_size, batch_size), desc="Validating"):
            val_predictions=apply_model(model, val_train[i:i+batch_size], val_train_Y, val_test[i:i+batch_size])
            
            val_prediction_view = val_predictions.reshape(-1, predicted_classes)
            val_target_view = torch.argmax(val_sev_test[i:i+batch_size], dim=-1).reshape(-1)

            loss_val = criterion(val_prediction_view, val_target_view)
            val_loss.append(loss_val.item())

            y_pred.append(val_prediction_view.cpu().numpy())
            y_true.append(val_target_view.cpu().numpy())

            _, predicted = torch.max(val_prediction_view, dim=-1)
            correct_classes=val_target_view.flatten()
            correct_predictions=(predicted==correct_classes).float()
            val_acc.append(correct_predictions.mean().item())

            for j in range(predicted_classes):
                class_mask=(correct_classes==j).float()
                val_sub_right[j]+= (correct_predictions * class_mask).sum().item()
                val_sub_total[j]+= class_mask.sum().item()


        y_pred=np.concatenate(y_pred, axis=0)
        y_true=np.concatenate(y_true, axis=0)

        val_loss=np.mean(val_loss)
        val_acc=np.mean(val_acc)
        val_sub_acc=[-1.0 for _ in range(predicted_classes)]
        for j in range(predicted_classes):
            if val_sub_total[j]>0:
                val_sub_acc[j]=val_sub_right[j]/val_sub_total[j]
        balanced_val_acc=np.mean(val_sub_acc)

    
    model.decoder_dict[head_name].train()
    # -----------------------
    val_time=time()-t0

    train_loss=np.mean(losses)

    total_time=data_time+train_time+val_time

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Bacc: {balanced_val_acc:.4f}')
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_balanced_acc": balanced_val_acc,
        "z_data_time": data_time,
        "z_train_time": train_time,
        "z_val_time": val_time,
        "total_time": total_time,
        "z_lr": scheduler.get_last_lr()[0],
    })
    wandb.log({f"x_val_sub_accuracy_{j}": val_sub_acc[j] for j in range(predicted_classes)})
    wandb.log({f"y_val_sub_total_{j}": val_sub_total[j] for j in range(predicted_classes)})

    save_model(model, os.path.join(save_path, "last.pth"))
    if val_loss < best_val_loss:
        best_val_loss=val_loss
        save_model(model, os.path.join(save_path, "best_loss.pth"))
    if train_loss < best_train_loss:
        best_train_loss=train_loss
        save_model(model, os.path.join(save_path, "best_train.pth"))
    if val_acc > best_val_acc:
        best_val_acc=val_acc
        save_model(model, os.path.join(save_path, "best_acc.pth"))
    if balanced_val_acc > best_val_bacc:
        best_val_bacc=balanced_val_acc
        save_model(model, os.path.join(save_path, "best_bacc.pth"))






wandb.finish()





