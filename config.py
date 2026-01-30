#Transformer embedding size
emsize=256

#number of attention heads
nhead=4

#number of hidden units in the feedforward network
nhid=512

#number of transformer layers
nlayers=4

#dropout rate
dropout=0.0

#length of each input sequence. Amount of data used for simulated training runs
seq_len=5000

#maximum number of features the model can handle
num_features=100

#allows for style encoding. Disabled for now
style_encoder=None

#normalize each input during training
input_normalization=False

#number of fomo head prediction (normal+abnormal)
n_out=2

#decoder dict describing just the default fomo head
decoder_dict={"standard": (None, n_out)}

#use special initializer for the default fomo head?
initializer=None

#whether to use efficient eval masking during inference
efficient_eval_masking=True

#no special decoder heads for now
decoder_once_dict={}

#no global attention tokens for now
num_global_att_tokens=0

#extra arguemnts to be passed to the model
model_extra_args={}

#unused during prediction
num_R=500
model_extra_args["model_para_dict"]={"num_R":num_R}

#use up to maximum number of features when generating training data
max_feature_dim=num_features

#how many classes to predict
num_class=n_out

#data  preprocessing transformation. Quantile makes data more gaussian/similar to training data
preprocess_transform="quantile"# alternative: set to "none" for there to be no quantile transformation
