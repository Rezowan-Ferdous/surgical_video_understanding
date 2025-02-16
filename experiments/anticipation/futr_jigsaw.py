import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import math
import argparse
from opts import parser
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import argparse
import pdb
import random
from torch.backends import cudnn
# from experiments.anticipation.anticipation_opts import parser
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.action_anticipation.futr import FUTR
from trainers.anticipation_trainer import train
from trainers.anticipation_trainer import predict
from data.common.baseanticipationdataset import BaseAnticipationDataset,AnticipationBase
from data.jigsaw.dataloader import JigsawAnticipation 
from data.jigsaw.preprocessing import jigsaw_files_df,actiondict,make_full_df

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


from trainers.anticipation_trainer import train
from trainers.anticipation_trainer import predict

sample_rate= 1

base_path = '/Users/khoedjarahman/Dropbox/Datasets/jigsaw/jigsaw/jigsaw_features'
jigsaw_root=base_path

root_folder ='.'
import pandas as pd
num_classes= len(actiondict)




csv_filename= str(sample_rate)+'_jigsaw.csv'
ds_dir= 'data/jigsaw'
# # Check if the file exists
df_dir_path= os.path.join(os.getcwd(),ds_dir)
df_file_path= os.path.join(df_dir_path,csv_filename)
if os.path.exists(df_file_path):
#     # Load the DataFrame from the file
    jigsaw_df = pd.read_csv(df_file_path)
    print(f"Loaded DataFrame from {df_file_path}:")
else:
    os.makedirs(df_dir_path,exist_ok=True)
#     # If file doesn't exist, create a new DataFrame
    print(f"{df_file_path} not found, creating a new DataFrame.")
    jigsaw_files_df = jigsaw_files_df(jigsaw_root)
    jigsaw_df = make_full_df(jigsaw_files_df, actiondict, sample_rate)
    # extract_features(df=jigsaw_full_df,pretrained_model=pretrained_model,feature_reducer=feature_reducer,batch_size=320)


    jigsaw_df.to_csv(df_file_path, index=False)
    print(f"New DataFrame created and saved to {df_file_path}.")

batch_size=1

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available. Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU.")


print('jigsaw columns',jigsaw_df.columns)
#jigsaw_df.head())

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
                for base_lr in self.base_lrs
            ]



# sample = train_data[0]  # Fetch the first sample
# for key, value in sample.items():
#     print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

# import matplotlib.pyplot as plt

# features = sample['features'].numpy()
# labels = sample['past_label'].numpy()

# plt.plot(features.mean(axis=1), label="Feature Mean")  # Plot mean of each feature frame
# plt.plot(labels, label="Labels", linestyle="dashed")
# plt.legend()
# plt.show()
# batch = next(iter(train_loader))

# for key, value in zip(["features", "past_label", "trans_future_dur", "trans_future_target"], batch):
#     print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

# for i in range(len(train_data)):
#     item = train_data[i]
#     print(f"Sample {i}: Features Shape: {item['features'].shape}, Past Labels: {item['past_label'].shape}")
def main():
    args=parser.parse_args()

    print('runs:',args.runs)    
    print('model:',args.model)

    dataset=args.dataset
    task=args.task

    if dataset=='jigsaw':
        data_path='data/jigsaw'

    actions_dict=actiondict
    pad_idx=-100
    num_classes=len(actions_dict)
    model = FUTR(n_class=num_classes, hidden_dim=2048, n_head=args.n_head,src_pad_idx=pad_idx,n_query=args.n_query,num_encoder_layers=6,num_decoder_layers=6,args=args,device=device)
    print('model:',model)
    model_save_path=os.path.join(args.model_save_path,args.model)
    results_save_path=os.path.join(args.results_save_path,args.model)


    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, total_epochs=args.epochs)

    criterion = nn.MSELoss(reduction = 'none')
    args.predict=False
    args.device = device
    if args.predict:
        obs_perc=[0.2,0.3]

        predict(model, args, model_save_path, results_save_path, actions_dict, task, dataset, num_classes)
        for obs_p in obs_perc:
            model.load_state_dict(torch.load(model_save_path))
            model.to(args.device)
            predict(model, args, model_save_path, results_save_path, actions_dict, task, dataset, num_classes, obs_p)

    else:

        
        train_data = JigsawAnticipation(
                dataframe=jigsaw_df[:10],
                action_dict=actiondict,
                sample_rate=sample_rate,
            )


        train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True if batch_size > 1 else False,
                collate_fn=train_data.my_collate, 
                pin_memory=True
            )
        for i in range(len(train_data)):
            item = train_data[i]
            print(f"Sample {i}: Features Shape: {item['features'].shape}, Past Labels: {item['past_label'].shape}")
        print(f"Dataset size: {len(train_data)}")

        train(args=args,model=model, train_loader=train_loader, optimizer=optimizer,scheduler=scheduler, criterion=criterion,  model_save_path=model_save_path, pad_idx=pad_idx, device=device )


if __name__ == '__main__':
    main()

# export PYTHONPATH="/home/local/data/rezowan/experiments/surgical_video_understanding:$PYTHONPATH"
# source ~/.bashrc