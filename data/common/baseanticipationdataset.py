import torch
import numpy as np
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from numpy.random import randint
import torch.nn.functional as F
# from utils import *
import pdb
import random


class BaseAnticipationDataset(Dataset):
    def __init__(self, vid_list, actions_dict, features_path, gt_path, pad_idx, n_class,
                 n_query=8,  mode='train', obs_perc=0.2, args=None):
        self.n_class = n_class
        self.actions_dict = actions_dict
        self.pad_idx = pad_idx
        self.features_path = features_path
        self.gt_path = gt_path
        self.mode = mode
        self.sample_rate = args.sample_rate
        self.vid_list = list()
        self.n_query = n_query
        self.args = args
        self.NONE = self.n_class - 1

        if self.mode == 'train' or self.mode == 'val':
            for vid in vid_list:
                self.vid_list.append([vid, .2])
                self.vid_list.append([vid, .3])
                self.vid_list.append([vid, .5])
        elif self.mode == 'test' :
            for vid in vid_list:
                self.vid_list.append([vid, obs_perc])

        self._make_input(vid, 0.2)


    def __getitem__(self, idx):
        vid_file, obs_perc = self.vid_list[idx]
        obs_perc = float(obs_perc)
        item = self._make_input(vid_file, obs_perc)
        return item


    def _make_input(self, vid_file, obs_perc ):
        vid_file = vid_file.split('/')[-1]
        vid_name = vid_file

        gt_file = os.path.join(self.gt_path, vid_file)
        feature_file = os.path.join(self.features_path, vid_file.split('.')[0]+'.npy')
        features = np.load(feature_file)
        features = features.transpose()

        file_ptr = open(gt_file, 'r')
        all_content = file_ptr.read().split('\n')[:-1]
        vid_len = len(all_content)
        observed_len = int(obs_perc*vid_len)
        pred_len = int(0.5*vid_len)

        start_frame = 0

        # feature slicing
        features = features[start_frame : start_frame + observed_len] #[S, C]
        features = features[::self.sample_rate]

        past_content = all_content[start_frame : start_frame + observed_len] #[S]
        past_content = past_content[::self.sample_rate]
        past_label = self.seq2idx(past_content)

        if np.shape(features)[0] != len(past_content) :
            features = features[:len(past_content),]

        future_content = \
        all_content[start_frame + observed_len: start_frame + observed_len + pred_len] #[T]
        future_content = future_content[::self.sample_rate]
        trans_future, trans_future_dur = self.seq2transcript(future_content)
        trans_future = np.append(trans_future, self.NONE)
        trans_future_target = trans_future #target


        # add padding for future input seq
        trans_seq_len = len(trans_future_target)
        diff = self.n_query - trans_seq_len
        if diff > 0 :
            tmp = np.ones(diff)*self.pad_idx
            trans_future_target = np.concatenate((trans_future_target, tmp))
            tmp_len = np.ones(diff+1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))
        elif diff < 0 :
            trans_future_target = trans_future_target[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else :
            tmp_len = np.ones(1)*self.pad_idx
            trans_future_dur = np.concatenate((trans_future_dur, tmp_len))


        item = {'features':torch.Tensor(features),
                'past_label':torch.Tensor(past_label),
                'trans_future_dur':torch.Tensor(trans_future_dur),
                'trans_future_target' : torch.Tensor(trans_future_target),
                }

        return item


    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''

        b_features = [item['features'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_dur = [item['trans_future_dur'] for item in batch]
        b_trans_future_target = [item['trans_future_target'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self.pad_idx)
        b_trans_future_dur = torch.nn.utils.rnn.pad_sequence(b_trans_future_dur, batch_first=True,
                                                        padding_value=self.pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self.pad_idx)

        batch = [b_features, b_past_label, b_trans_future_dur, b_trans_future_target]

        return batch


    def __len__(self):
        return len(self.vid_list)

    def seq2idx(self, seq):
        idx = np.zeros(len(seq))
        for i in range(len(seq)):
            idx[i] = self.actions_dict[seq[i]]
        return idx

    def seq2transcript(self, seq):
        transcript_action = []
        transcript_dur = []
        action = seq[0]
        transcript_action.append(self.actions_dict[action])
        last_i = 0
        for i in range(len(seq)):
            if action != seq[i]:
                action = seq[i]
                transcript_action.append(self.actions_dict[action])
                duration = (i-last_i)/len(seq)
                last_i = i
                transcript_dur.append(duration)
        duration = (len(seq)-last_i)/len(seq)
        transcript_dur.append(duration)
        return np.array(transcript_action), np.array(transcript_dur)



class AnticipationBase(Dataset):
    def __init__(self,actions_dict,dataframe=None, mode='train', sample_rate=1,obs_perc=0.2,args=None):
        super().__init__()
        self.mode=mode
        self.pad_idx=args.pad_idx if args else 0
        self.n_class=args.num_classes if args else 10
        self.NONE =self.n_class-1
        self.actions_dict = actions_dict
        self.n_query= args.n_query if args else 8

        if dataframe is not None:
            self.data=self._process_dataframe(dataframe,obs_perc)
        else:
            raise ValueError("DataFrame input is required for AnticipationBase")

        self.sample_rate=sample_rate
        self.obs_perc=obs_perc
        self.args=args
            # self.vid_list=self.dataframe['video'].unique()
            # self.actions_dict=self.dataframe['action'].unique()
            # self.features_path=args.features_path
            # self.gt_path=args.gt_path
            # self.dataset=BaseAnticipationDataset(vid_list=self.vid_list, actions_dict=self.actions_dict,
            #                                      features_path=self.features_path, gt_path=self.gt_path,
            #                                      pad_idx=self.pad_idx, n_class=self.n_class,
            #                                      n_query=self.n_query, mode=self.mode, obs_perc=self.obs_perc, args=self.args)

    def _process_dataframe(self,dataframe,obs_perc):
        data= []
        for _,row in dataframe.iterrows():
            vid=row['video_id']
            feature_path=row['feature_path']
            annotation_path=row['annotation_path']
            obs_perc=[0.2,0.3,0.5] if self.mode in ['train','val'] else [obs_perc]
            
            for obs in obs_perc:
                data.append({
                    "video_id":vid,
                    "feature_path":feature_path,
                    "annotation_path":annotation_path,
                    "labels":row['labels'],
                    "segments":row['segments'],
                    "frames":row['frames'],
                    "obs_perc":obs
                })

        return data
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self,idx):
        item_data=self.data[idx]
        return self._make_input(item_data,sample_rate=self.sample_rate)
    

    def _make_input(self,item_data,sample_rate=1):
        feature_file=item_data['feature_path']
        gt_file=item_data['annotation_path']
        obs_perc=item_data['obs_perc']
        str_labels=item_data['labels']
        all_labels= np.load(str_labels)
        # int_list = list(map(int, str_labels.strip('[]').split()))
        # all_labels = np.array(int_list)
        # print('all labels',all_labels)

        features=np.load(feature_file) # [T, C]
        # print('features',features.shape)
        # **Efficient Resampling Instead of Simple Slicing**
        if sample_rate > 1:
            original_length = len(features)
            new_len= original_length//sample_rate
            features = F.interpolate(features, size=new_len, mode="linear", align_corners=False).squeeze(0)
            features = features.numpy()
        
        # all_labels = [all_labels[int(i * sample_rate)] for i in range(new_len)]  # Resample labels too

        # features=features[::sample_rate]

        # with open(gt_file, 'r') as f:
        #     all_labels = f.read().strip().split('\n')

        vid_len=min(len(features),len(all_labels))
        features,all_labels=features[:vid_len],all_labels[:vid_len]


        observed_len = int(obs_perc * vid_len)  # Compute observed length
        pred_len = int(0.5 * vid_len)  # Predict half the video

        # **Sanity check: Feature & label lengths must match**
        if len(features) != len(all_labels):
            raise ValueError(
                f"Mismatch: Features ({len(features)}) vs Labels ({len(all_labels)}) "
                f"in {feature_file}. Check sample_rate settings."
            )
        # **Extract observed portion**
        past_features = features[:observed_len]
        # print('all_labels',all_labels)
        # print('observed_len - all_labels ',observed_len,len(all_labels), all_labels[:observed_len])
        # past_labels = self.seq2idx(all_labels[:observed_len])
        past_labels = all_labels[:observed_len]

        # **Extract future transcript**
        future_labels = all_labels[observed_len : observed_len + pred_len]
        trans_future, trans_future_dur = self.seq2transcript(future_labels)
        trans_future = np.append(trans_future, self.NONE)

        # **Pad future sequence if needed**
        diff = self.n_query - len(trans_future)
        if diff > 0:
            trans_future = np.concatenate((trans_future, np.ones(diff) * self.pad_idx))
            trans_future_dur = np.concatenate((trans_future_dur, np.ones(diff + 1) * self.pad_idx))
        elif diff < 0:
            trans_future = trans_future[:self.n_query]
            trans_future_dur = trans_future_dur[:self.n_query]
        else:
            trans_future_dur = np.concatenate((trans_future_dur, np.ones(1) * self.pad_idx))

        return {
            'features': torch.Tensor(past_features),
            'past_label': torch.Tensor(past_labels),
            'trans_future_dur': torch.Tensor(trans_future_dur),
            'trans_future_target': torch.Tensor(trans_future),
        }
    
    def my_collate(self, batch):
        """Custom collate function for batching."""
        keys = ['features', 'past_label', 'trans_future_dur', 'trans_future_target']
        collated = {key: torch.nn.utils.rnn.pad_sequence([item[key] for item in batch], 
                                                          batch_first=True, padding_value=self.pad_idx) 
                    for key in keys}
        return [collated[key] for key in keys]


    def seq2idx(self, seq):
        """Convert sequence labels to indices. If seq is already indices, return as-is."""
        # print('seq', seq)
        if isinstance(seq[0], str):  # If labels are strings, map them using actions_dict
            return np.array([self.actions_dict[action] for action in seq], dtype=np.float32)
        else:  # If labels are already indices, return them directly
            return np.array(seq, dtype=np.float32)
        
    def seq2transcript(self, seq):
        """Convert sequence to action transcript and duration."""
        if not seq.size:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        # If seq contains string labels, map them to indices first
        if isinstance(seq[0], str):
            seq = np.array([self.actions_dict[action] for action in seq], dtype=np.float32)

        transcript_action = [seq[0]]  # Start with the first action
        transcript_dur = []
        last_i = 0

        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:  # If action changes
                transcript_action.append(seq[i])  # Add the new action
                transcript_dur.append((i - last_i) / len(seq))  # Add duration
                last_i = i

        # Add the duration of the last action
        transcript_dur.append((len(seq) - last_i) / len(seq))
        return np.array(transcript_action, dtype=np.float32), np.array(transcript_dur, dtype=np.float32)