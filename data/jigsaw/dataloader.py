import os,torch
from tqdm import tqdm
import torch
import torch.nn as nn
import cv2
import os
import pandas as pd
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from typing import Any, Dict, List, Optional
import torch.nn.functional as F
from data.common.baseanticipationdataset import BaseAnticipationDataset, AnticipationBase
from jigsaw.preprocessing import jigsaw_files_df,actiondict,make_full_df


# base_path = '/Users/khoedjarahman/Dropbox/Datasets/jigsaw/jigsaw/jigsaw_features'
# jigsaw_root=base_path
# jigsawdf= jigsaw_files_df(jigsaw_root)    

# fulldf = make_full_df(jigsawdf)


class JigsawDataset(Dataset):


    def __init__(self, dataframe, root, num_classes, action_dict, sample_rate=1):
        self.df = dataframe
        self.root = root
        self.num_class = num_classes
        self.action_dict = action_dict
        self.sample_rate = sample_rate


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx % len(self.df)]
        feat_file, annot_file, labels, boundaries, segments, frames, vid = (
            row['feature_path'], row['annotation_path'], row['labels'], row['boundaries'], row['segments'],
            row['frames'], row['video_id'])
        features = np.load(feat_file).astype(np.float32)
        max_index = features.shape[0]
        labels = np.load(labels)
        boundaries = np.load(boundaries)
        indices = np.linspace(0, max_index - 1, labels.shape[0]).astype(int)
        #
        # # Sample the sequence, label, and mask using the generated indices
        features = features[indices, :]  # Select along the time dimension


        # Ensure that all arrays have the same length initially
        assert features.shape[0] == labels.shape[0] == boundaries.shape[0], "Initial lengths are not the same!"


        # # Sample the features, labels, and boundaries with the same rate
        # sampled_features = feature[::self.sample_rate, :]  # Sample along the first dimension
        # sampled_labels = labels[::self.sample_rate]  # Sample the labels
        # sampled_boundaries = boundaries[::self.sample_rate]  # Sample the boundaries


        sample = {
            "feature": features,
            "label": labels,
            "feature_path": feat_file,
            "boundary": boundaries,
        }
        return sample


    def resampled_data(self, num_frames, fps, annotation_path, feature_file, sample_rate):
        with open(annotation_path, 'r') as file:
            lines = file.readlines()


        segments = []
        total_frames = num_frames
        lastframe = 0


        # Initialize arrays for tracking sampled frames across segments
        reduced_labels = []
        reduced_boundaries = []


        for line in lines:
            # Split line by ',' and extract the start_frame, end_frame, and action
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue  # Skip malformed lines


            start_frame, end_frame, action = parts
            try:
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                action = int(action)  # Assuming action is an integer


                # Ensure the end_frame does not exceed total number of frames
                end_frame = min(end_frame, total_frames - 1)


                # Apply sampling rate within the segment
                sampled_frames = np.arange(start_frame, end_frame + 1, sample_rate)


                # Append the action to reduced_labels for sampled frames
                reduced_labels.extend([action] * len(sampled_frames))


                # Handle boundary assignment for sampled frames
                sampled_length = len(sampled_frames)


                if sampled_length >= 10:  # If the segment is long enough
                    # Start boundary: 1, Middle: 0, End boundary: 2
                    reduced_boundaries.extend([1] * min(5, sampled_length))  # Start boundary
                    reduced_boundaries.extend([0] * (sampled_length - 10))  # Inside action
                    reduced_boundaries.extend([2] * min(5, sampled_length))  # End boundary
                else:
                    # Short segment: Mark entire segment with boundary
                    reduced_boundaries.extend([1] * (sampled_length - 1))  # Start and mid boundary
                    reduced_boundaries.append(2)  # End boundary


            except ValueError:
                continue  # Skip lines with invalid data


            frame_length = end_frame - start_frame
            duration = frame_length / fps


            # Store segment information
            segments.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'label': action,
                'duration': duration
            })


        # After processing all segments, convert lists to numpy arrays
        labels = np.array(reduced_labels)
        boundaries = np.array(reduced_boundaries)


def collate_fn(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max([s["feature"].shape[0] for s in sample])


    feat_list = []
    label_list = []
    path_list = []
    boundary_list = []
    length_list = []


    for s in sample:
        feature = s["feature"]
        label = s["label"]
        boundary = s["boundary"]
        feature_path = s["feature_path"]


        feature = feature.T
        _, t = feature.shape
        pad_t = max_length - t


        length_list.append(t)


        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label)
        boundary = torch.from_numpy(boundary)
        # print("shape length",t)
        if pad_t > 0:
            feature = F.pad(feature, (0, pad_t), mode="constant", value=0.0)
            label = F.pad(label, (0, pad_t), mode="constant", value=255)
            boundary = F.pad(boundary, (0, pad_t), mode="constant", value=0.0)


        # reshape boundary (T) => (1, T)  / boundary.unsqueeze(0)
        # boundary = torch.from_numpy(boundary).unsqueeze(0)


        # label= torch.from_numpy(label).unsqueeze(0)
        feat_list.append(feature)
        label_list.append(label)
        path_list.append(feature_path)
        # boundary_list.append(boundary)
        boundary_list.append(boundary)


    # print(feat_list)
    # merge features from tuple of 2D tensor to 3D tensor
    features = torch.stack(feat_list, dim=0)
    # merge labels from tuple of 1D tensor to 2D tensor
    labels = torch.stack(label_list, dim=0)


    # merge labels from tuple of 2D tensor to 3D tensor
    # shape (N, 1, T)
    boundaries = torch.stack(boundary_list, dim=0)


    # generate masks which shows valid length for each video (N, 1, T)
    masks = [[1 if i < length else 0 for i in range(max_length)] for length in length_list]
    masks = torch.tensor(masks, dtype=torch.bool)
    # print("mask , feature labesl and boundary shape for training ",masks.shape, features.shape,labels.shape,boundaries.shape)


    return {
        "feature": features,
        "label": labels,
        "boundary": boundaries,
        "feature_path": path_list,
        "mask": masks,
    }

class JigsawAnticipation(AnticipationBase):
    def __init__(self, dataframe, action_dict, mode='train', sample_rate=1, args=None):
        super(JigsawAnticipation, self).__init__(actions_dict=action_dict, dataframe=dataframe, mode=mode, sample_rate=sample_rate, args=args)
        self.sample_rate = sample_rate

    def __getitem__(self, idx):
        item_data = self.data[idx]
        return self._make_input(item_data, sample_rate=self.sample_rate)
