import os,random,sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Any, Dict, List, Optional


import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torch
import torch.nn.functional as F




from utils.video_preprocess import get_video_prop

rarp50= {'train1': [2, 4, 5, 6, 8, 9, 10, 11, 12, 15, 16, 17, 19],
                'train2'  : [1, 3, 7, 13, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 33, 34, 35, 36, 37, 38, 39, 40],
                'test' : [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
            }


def feat_annot_files(dataset_dir,video_filename,feature_filename,annot_filename):
    video_file = video_filename #'video_left.avi'
    feature_file = feature_filename #'feat_2304.npy'
    annot_file = annot_filename #'action_continuous.txt'
    video_folders = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)]
    video_files = []
    feature_files = []
    annotation_files = []
    for folder in video_folders:
        vid_path = os.path.join(folder, video_file)
        feat_path = os.path.join(folder, feature_file)
        annot_path = os.path.join(folder, annot_file)
        if os.path.isfile(vid_path) and os.path.isfile(annot_path):
            video_files.append(vid_path)
            annotation_files.append(annot_path)
            feature_files.append(feat_path)
    return video_files, annotation_files,feature_files


def create_dataframe_from_annotations(file_pairs,fps=60):
    data = []
    # annotations = []


    for feature_path, annotation_path,video_path, video_id in file_pairs:
        # Read the annotation file
        video_prop = get_video_prop(video_path)
        # print(video_prop)
        vid_duration =video_prop['num_frames'] / video_prop['fps']
        resolution= (video_prop['height'],video_prop['width'])
        with open(annotation_path, 'r') as file:
            lines = file.readlines()


        segments = []
        total_frames=video_prop['num_frames']
        lastframe= 0
        # print('total_frames,lastframe',total_frames,lastframe)
        labels = np.full(total_frames, -100)  # Initialize with -100 for all frames
        boundaries = np.full(total_frames, -100)  # Initialize boundaries similarly


        for line in lines:
            # Split line by ', ' and extract the values
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue  # Skip malformed lines


            start_frame, end_frame, action = parts
            try:
                # print(start_frame, end_frame, action)
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                action = int(action)  # Assuming action is an integer
                # Ensure the end_frame does not exceed the total number of frames
                end_frame = min(end_frame, total_frames - 1)


                # Assign the action to all frames in the range start_frame:end_frame (inclusive)
                labels[start_frame:end_frame + 1] = action


                # labels[start_frame:end_frame] = action
                lastframe= end_frame
                # Handle boundary assignment for short segments
                segment_length = end_frame - start_frame + 1


                if segment_length >= 10:  # If the segment is long enough
                    # Define boundary regions (start boundary, inside, end boundary)
                    boundaries[start_frame:start_frame + 5] = 1  # Start boundary
                    boundaries[start_frame + 5:end_frame - 5] = 0  # Inside action
                    boundaries[end_frame - 5:end_frame + 1] = 2  # End boundary
                else:
                    # For short segments (< 10 frames), mark the whole region with boundary 1 (start/end)
                    boundaries[start_frame:end_frame + 1] = 1  # Treat the whole region as boundary
                    boundaries[start_frame + 1:end_frame] = 2  # Mark last frame(s) as end boundary


            except ValueError:
                continue  # Skip lines with invalid data


            frame_length = end_frame - start_frame
            duration = frame_length/fps


            segments.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'label': action,
                'duration': duration
            })


        save_pathl = os.path.join(os.path.dirname(feature_path), 'labels.npy')
        save_pathb = os.path.join(os.path.dirname(feature_path), 'boundaries.npy')


        if os.path.exists(os.path.join(os.path.dirname(feature_path), video_id+'labels.npy')):
            os.remove(os.path.join(os.path.dirname(feature_path), video_id+'labels.npy'))
        if os.path.exists(os.path.join(os.path.dirname(feature_path), video_id+'boundaries.npy')):
            os.remove(os.path.join(os.path.dirname(feature_path), video_id + 'boundaries.npy'))


        labels_cnt= np.unique(labels,return_counts=True)
        bound_cnt =np.unique(boundaries,return_counts=True)
        # print('lastframe,len(labels)',lastframe,len(labels),labels_cnt ,bound_cnt ,len(segments))
        # # Append the data for the current video
        data.append({
        'video_id': video_id,
        'feature_path': feature_path,
        'video_path':video_path,
        'annotation_path':annotation_path,
        'segments': segments,
        'duration': vid_duration,
        'fps': video_prop['fps'],
        'resolution':resolution,
        'frames': video_prop['num_frames'],
        'labels':save_pathl,
         'boundaries': save_pathb
        })


        np.save(save_pathl,labels)
        np.save(save_pathb, boundaries)


    # Create DataFrame
    # df = pd.DataFrame(annotations)
    df = pd.DataFrame(data)
    return df


def validate_and_pair_files(video_paths, label_paths,feature_path):
    # Find matching files
    matched_pairs = []
    for video_path, label_path,feat_path in zip(video_paths, label_paths,feature_path):
        vid_base_name = os.path.basename(os.path.dirname(video_path))
        label_base_name = os.path.basename(os.path.dirname(label_path))
        if vid_base_name == label_base_name:
            matched_pairs.append((video_path, label_path, feat_path,vid_base_name))
        else:
            print(f"Warning: No matching annotation file for video {video_path}")
    if len(matched_pairs) != len(video_paths):
        raise ValueError("Not all video files have corresponding annotation files.")
    return matched_pairs
@staticmethod
def create_dataframes(dataset_dir,video_filename,feature_filename,annot_filename):
    feature_files, annotation_files,video_files = feat_annot_files(dataset_dir,video_filename,feature_filename,annot_filename)
    filepairs = validate_and_pair_files(video_files, annotation_files,feature_files)
    df = create_dataframe_from_annotations(filepairs)
    return df






class RARPDataset(Dataset):
    def __init__(self,dataframe,root,num_classes,action_dict,sample_rate=1):
        self.df=dataframe
        self.root=root
        self.num_class=num_classes
        self.action_dict= action_dict
        self.sample_rate=sample_rate


    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx % len(self.df)]
        feat_file,annot_file,label_file,boundary_file,segments,frames,vid,fps =(
            row['feature_path'],row['annotation_path'],row['labels'],row['boundaries'],row['segments'],row['frames'],row['video_id'],row['fps'])
        feature = np.load(feat_file).astype(np.float32)
        label = np.load(label_file).astype(np.int64)
        boundary = np.load(boundary_file).astype(np.float32)


        # Ensure that all arrays have the same length initially
        assert feature.shape[0] == label.shape[0] == boundary.shape[0], "Initial lengths are not the same!"




        # Sample the features, labels, and boundaries with the same rate
        sampled_features = feature[::self.sample_rate, :]  # Sample along the first dimension
        sampled_labels = label[::self.sample_rate]  # Sample the labels
        sampled_boundaries = boundary[::self.sample_rate]  # Sample the boundaries


        # Ensure the shapes match after sampling
        assert sampled_features.shape[0] == sampled_labels.shape[0] == sampled_boundaries.shape[0], "Lengths after sampling do not match!"


        # if self.transform is not None:
        #     feature, label, boundary = self.transform([feature, label, boundary])


        sample = {
            "feature": sampled_features,
            "label": sampled_labels,
            "feature_path": feat_file,
            "boundary": sampled_boundaries,
        }
        return sample


    def resampled_data(self,num_frames, fps, annotation_path, feature_file, sample_rate):
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


        feature= torch.from_numpy(feature)
        label= torch.from_numpy(label)
        boundary=torch.from_numpy(boundary)
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




class RARPBatchGenerator(object):
    def __init__(self, dataframe, num_classes, actions_dict, sample_rate=1):


        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.sample_rate = sample_rate
        self.df = dataframe
        self.list_of_examples = []
        self.index = 0
        # print(dataframe)
        self.read_data()


    def reset(self):
        """Reset the index and shuffle the examples."""
        self.index = 0
        random.shuffle(self.list_of_examples)


    def has_next(self):
        """Check if there are more examples left in the dataset."""
        return self.index < len(self.list_of_examples)


    def read_data(self):
        """Read data from the provided dataframe."""
        # Convert dataframe rows to a list of tuples (feature_file, annotation_file)
        # print(self.df)
        self.list_of_examples = list(self.df[['feature_path', 'annotation_path']].itertuples(index=False, name=None))
        random.shuffle(self.list_of_examples)


    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)


    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()


        warped_batch_input_tensor = self.timewarp_layer(batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(batch_target_tensor, grid,
                                                         mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(1).long()  # obtain the same shape


        return warped_batch_input_tensor, warped_batch_target_tensor


    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''


        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features


        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))


    def next_batch(self, batch_size, if_warp=False):
        """Generate the next batch of data."""
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size


        batch_input = []
        batch_target = []
        batch_position = []
        for feature_file, annotation_file in batch:
            features = np.load(feature_file)
            with open(annotation_file, 'r') as file_ptr:
                content = file_ptr.read().split('\n')[:-1]


            # Initialize classes array with zeros and populate it according to the annotation
            # classes = np.zeros(min(np.shape(features)[0], len(content)))
            num_frames = features.shape[0]
            classes = np.full(num_frames, -100)  # Initialize with -100 for padding
            positions = np.full(num_frames, -100)
            # print(classes.shape)
            # positions = np.zeros(np.shape(features)[0])
            for line in content:
                start, end, action = map(int, line.split(','))
                # print(start,end,action)
                classes[start:end + 1] = self.actions_dict[action]
                # Assign position labels:
                # First 5 frames of the segment get a position label of 1
                positions[start:start + 5] = 1
                positions[6:(end - 5)] = 3
                # Last 5 frames of the segment get a position label of 2
                positions[end - 4:end + 1] = 2


            # Subsample the features and classes according to the sample rate
            batch_input.append(features[::self.sample_rate, :])
            batch_target.append(classes[::self.sample_rate])
            batch_position.append(positions[::self.sample_rate])


        # Determine the maximum sequence length in the batch
        length_of_sequences = list(map(len, batch_target))


        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[1], max(length_of_sequences),
                                         dtype=torch.float)


        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        batch_position_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        boundary_mask = torch.zeros(len(batch_input), 3, max(length_of_sequences), dtype=torch.float)


        # print(f'length_of_sequences {length_of_sequences}, np.shape(batch_input[0])[1]{np.shape(batch_input[0])[1]}, '
        #       f'batch_input_tensor.shape {batch_input_tensor.shape} , batch_target_tensor {batch_target_tensor.shape}, mask {mask.shape} ')
        # # Fill in the tensors with the batch data
        # print('batch input ', len(batch_input))


        for i in range(len(batch_input)):
            # print('batch input inside shape ', batch_input[i].shape)
            # print('batch input inside shape[0] ', batch_input[i].shape[0])
            # print('batch target inside shape[0] ', np.shape(batch_target[i])[0])
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0),
                                                              torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i,
                                                                        :np.shape(batch_target[i])[
                                                                            0]] = warped_input.squeeze(
                    0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[0]] = torch.from_numpy(batch_input[i]).T
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
            boundary_mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(3, np.shape(batch_position[i])[0])


        # for i in range(len(batch_input)):
        #     batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
        #     # Transpose batch_input[i] to match the expected shape
        #     # batch_input_tensor[i, :, :np.shape(batch_input[i])[0]] = torch.from_numpy(batch_input[i].T)
        #     batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
        #     batch_position_tensor[i, :np.shape(batch_position[i])[0]] = torch.from_numpy(batch_position[i])
        #     mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        # print(batch_input_tensor.shape, batch_target_tensor.shape, batch_position_tensor.shape, mask.shape,boundary_mask.shape )
        return batch_input_tensor, batch_target_tensor, batch_position_tensor, mask, boundary_mask, batch




import numpy as np
from scipy.stats import truncnorm
import torch.nn.functional as TF
import torch.nn as nn


class TimeWarpLayer(nn.Module):
    def __init__(self):
        super(TimeWarpLayer, self).__init__()


    def forward(self, x, grid, mode='bilinear'):
        '''
        :type&shape x: (cuda.)FloatTensor, (N, D, T)
        :type&shape grid: (cuda.)FloatTensor, (N, T, 2)
        :type&mode: bilinear or nearest
        :rtype&shape: (cuda.)FloatTensor, (N, D, T)
        '''
        assert len(x.shape) == 3
        assert len(grid.shape) == 3
        assert grid.shape[-1] == 2
        x_4dviews = list(x.shape[:2]) + [1] + list(x.shape[2:])
        grid_4dviews = list(grid.shape[:1]) + [1] + list(grid.shape[1:])
        out = TF.grid_sample(input=x.view(x_4dviews), grid=grid.view(grid_4dviews), mode=mode, align_corners=True).view(x.shape)
        return out




class GridSampler():
    def __init__(self, N_grid, low=1, high=5):  # high=5
        N_primary = 100 * N_grid
        assert N_primary % N_grid == 0
        self.N_grid = N_grid
        self.N_primary = N_primary
        self.low = low
        self.high = high


    def sample(self, batchsize=1):
        num_centers = np.random.randint(low=self.low, high=self.high)
        lower, upper = 0, 1
        mu, sigma = np.random.rand(num_centers), 1 / (num_centers * 1.5)  # * 1.5
        TN = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        vals = TN.rvs(size=(self.N_primary, num_centers))
        grid = np.sort(
            np.random.choice(vals.reshape(-1), size=self.N_primary, replace=False))  # pick one center for each primary
        grid = (grid[::int(self.N_primary / self.N_grid)] * 2 - 1).reshape(1, self.N_grid, 1)  # range [-1, 1)
        grid = np.tile(grid, (batchsize, 1, 1))
        grid = np.concatenate([grid, np.zeros_like(grid)], axis=-1)
        return grid