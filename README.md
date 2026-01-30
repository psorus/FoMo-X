#FoMo-X 
Source code for our paper proposing multi-headed foundation models for anomaly detection explainability.

##Training
Both heads included in our paper are trained using `train_severity_head.py` and `train_variation_head.py` respectively. These require artificial data from `generate_data_severity.py` and `generate_data_variation.py` to train on.

This requires a pretrained outlier foundation model (`original_fomo.ckpt`) and adds one additional head to this model. `assemble.py` combines these one plus two heads into a singular model (`fomox.ckpt`).

##Evaluation
Evaluation is done using `evaluate.py`. For this please clone ADBench (`https://github.com/Minqi824/ADBench.git`) into this folder and run `python3 evaluate.py`. This will save the predictions into results.npz.

##Prediction
To output the prediction of FoMo-X please import `predict(train,test)` from `predict.py`

##Requirements
- numpy
- torch
- tqdm
- scikit-learn
- scipy
Install all of these with `pip install -r requirements.txt`

