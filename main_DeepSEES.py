from vrae.vraeKmeansMSD import VRAE
from vrae.utils import *
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection  import train_test_split
import lstm_vae.utils as myutils
from lstm_vae.VraeAnalyzer import VraeAnalyzer


#%%
dload = './model_dir' #download directory
###data loader
path = './data/review/'
filename_set = ['Test_iso']
model_name = 'vrae_' + ''.join([i for i in filename_set[0]])\
              +'.pth'

#%% Hyper parameters
hidden_size = 128#128
hidden_layer_depth = 2
latent_length = 20 #latent dimension
batch_size = 50 #Normal diffusion with four HMM states.
learning_rate = 0.0005#0.0005
n_epochs = 40 # 40
dropout_rate = 0.2 #0.2
n_scale = 3 # scaling factor
optimizer = 'Adam' # options: ADAM, SGD
cuda = True # options: True, False
print_every= 500
clip = True # options: True, False
max_grad_norm=5 # 5
loss = 'MSELoss' # options: SmoothL1Loss, MSELoss
block = 'LSTM' # options:  LSTM, GRU
num_k = 10

#%%
is_train = True
num_features= 2
is_range = False % whether every dimension normalize
# if every dimension is dependent use this item
n_cluster= 2
#%%
seq_len= 30
# brownian_filter=(0.2,0.15) #(delta, model_error) (0.2 0.075)
brownian_filter=(0,-np.inf) #(delta, model_error)
alpha_ROI_select=(-np.inf,np.inf)#(0,0.5)#(-np.inf,np.inf)
# Reconstruct_error_select=(0,1) #(0,1)
Reconstruct_error_select=(0,1) #(0,1)
seed = 1 #0 seed = 2 for 15min-1 #128 for PDL_1? 128/1 for A5491_iso/re 1
 # if every dimension is dependent use this item

df,_,_= myutils.load_data(path,filename_set)
data,_ = myutils.data_processing(df,num_features,seq_len)

if is_range:
    scalar = n_scale/np.mean(np.mean(np.abs(data),axis=0),axis=0)
else:
    s = n_scale / np.mean(np.mean(np.abs(data)))
    scalar = [s for i in range(num_features)]
data = data * scalar
label = df.iloc[:, -1].values
X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.20, random_state=0)

num_classes = len(np.unique(y_train))
base = np.min(y_train)  # Check if data is 0-based
if base != 0:
    y_train -= base
y_val -= base
train_dataset = TensorDataset(torch.from_numpy(X_train))
test_dataset = TensorDataset(torch.from_numpy(X_val))
sequence_length = X_train.shape[1]
number_of_features = X_train.shape[2]
vrae = VRAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload,
            K = num_k,
            kmeans_weight = 5e-2,
            n_s= 4)
if is_train:
    vrae.fit(train_dataset)
    vrae.evaluate(test_dataset)

    vrae.save(model_name)

#%%
#%% VraeAnalyze load
vrae.load(dload + '/'+model_name)
Analyzer = VraeAnalyzer(vrae,path,filename_set,num_features=num_features,seq_len=seq_len,n_cluster=n_cluster,
             brownian_filter=brownian_filter,alpha_ROI_select=alpha_ROI_select,Reconstruct_error_select=Reconstruct_error_select,
             seed = seed, is_range = is_range, n_scale = n_scale, n_seg=None)

Analyzer.plot_filtering_traj(num=100,is_filtering=False,scalar = scalar) # seed = 2 of 15min-1, num=250,
Analyzer.run_centers(scalar = scalar)


#%%
Analyzer.plot_every_trajectory(is_filtering=True)
# Analyzer.plot_every_trajectory(is_filtering=True,lo=0,step=200)
# Analyzer.plot_every_time_series(lo=0,step=200,extend=0.05,sz=0.05) #0.05, 0.05
