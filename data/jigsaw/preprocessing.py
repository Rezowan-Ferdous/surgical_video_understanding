
# from utils.video_preprocess import ReducedKernelConv,extract_features_dinov2,extract_features

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



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# feature_reducer = ReducedKernelConv(device=device)
# feature_reducer = nn.DataParallel(feature_reducer)
# # dataframe = create_dataframes(base_test_dir,video_filename,feature_filename,annot_filename)


# pretrained_model = extract_features_dinov2




import albumentations as A
from albumentations.pytorch import ToTensorV2
# transform = A.Compose([
#     A.Resize(256, 256),
#     A.CenterCrop(224, 224),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])


# jigsaw_root= "/home/ubuntu/Dropbox/Datasets/jigsaw/jigsaw_dataset"
#
# jigsaw_path = "/home/local/data/rezowan/datasets/jigsaw"
from pathlib import Path
import os 
import numpy as np
import pandas as pd  
base_path = '/Users/khoedjarahman/Dropbox/Datasets/jigsaw/jigsaw/jigsaw_features'
jigsaw_root=base_path

actiondict={'G1' : 0 , #Reaching for needle with right hand
'G2': 1, # Positioning needle
'G3':2, # Pushing needle through tissue
'G4':3, #Transferring needle from left to right
'G5':4,# Moving to center with needle in grip
'G6':5,# Pulling suture with left hand
'G7':6 ,#Pulling suture with right hand
'G8':7 ,#Orienting needle
'G9':8 ,#Using right hand to help tighten suture
'G10':9 ,#Loosening more suture
'G11':10,# Dropping suture at end and moving to end points
'G12':11, #Reaching for needle with left hand
'G13': 12, #Making C loop around right hand
'G14':13,  #Reaching for suture with right hand
'G15':14, 
'NONE':-100 } #Pulling suture with both hands

def extract_video_identifier(video_file):
    return '_'.join(video_file.split('/')[-1].split('_')[:-1])
    # Split the filename and remove the capture part (e.g., Needle_Passing_B001 from Needle_Passing_B001_capture1)
    

# Function to extract the core identifier from the annotation file
def extract_annotation_identifier(annotation_file):
    # Get the annotation identifier (e.g., Needle_Passing_B001 from Needle_Passing_B001.txt)
    return annotation_file.split('/')[-1].replace('.txt', '')
    
def jigsaw_files_df(jigsaw_root):
    annot_files = []
    video_files = []
    
    for folders in os.listdir(jigsaw_root):
        pathname = os.path.join(jigsaw_root, folders)
        
        if os.path.isdir(pathname):
            for folder in os.listdir(pathname):
                folder_path = os.path.join(pathname, folder)
        
                if folder == 'transcriptions':
                    annotations = os.listdir(folder_path)
                    annotations.sort()
                    # Store absolute paths for annotation files
                    for annot in annotations:
                        annot_files.append(os.path.join(folder_path, annot))
        
                if folder == 'video':
                    videos = os.listdir(folder_path)
                    videos.sort()
                    # Store absolute paths for video files
                    for video in videos:
                        video_files.append(os.path.join(folder_path, video))
        else:
            print(f"{pathname} is not a directory.")



    # Create a mapping of annotation identifier to annotation file
    annotation_dict = {extract_annotation_identifier(annotation): annotation for annotation in annot_files}
    
    # For each video, find the corresponding annotation by the identifier without the capture part
    matched_annotations = []
    for video in video_files:
        video_identifier = extract_video_identifier(video)
        annotation = annotation_dict.get(video_identifier, None)
        matched_annotations.append(annotation)
    
    # Check the lengths of video_files and matched_annotations
    print(f"Length of video_files: {len(video_files)}")
    print(f"Length of matched_annotations: {len(matched_annotations)}")
    
    # Ensure both lists have the same length
    if len(video_files) == len(matched_annotations):
        # Create a DataFrame
        jigsawdf = pd.DataFrame({
            'video_path': video_files,  # Ensure the correct video list is used
            'annotation_path': matched_annotations
        })
        print(jigsawdf)
    else:
        print("Error: The lengths of video_files and matched_annotations do not match.")
    return jigsawdf

def read_jigsaw_annotation(annotation_path):
    with open(annotation_path, "r") as f:
        contents = f.read().split("\n")[:-1]
    return contents

def calculate_jigsaw_segments(annotation_path,actiondict,sample_rate):
    segments=[]
    contents= read_jigsaw_annotation(annotation_path)
    endf= int(contents[-1].strip().split()[1])

    cal_len = 0
    labels = []
    boundaries = []
    total_len=0
    if contents[0].split()[0]!=0:
        
        first=int(contents[0].split()[0])
        indc= np.arange(0, first-1, sample_rate)
        efl= len(indc)
        unlabeled= np.full(efl,-100)
        labels.extend(unlabeled)
        boundaries.extend(unlabeled)
        
    for content in contents:
        st, ed, act = content.split()
        start_frame = int(st)
        end_frame = int(ed) - 1  # End frame inclusive
        segment_length = end_frame - start_frame + 1  # Include the last frame
        
        label = actiondict[act]
        total_len+=segment_length
        
        
        # Calculate effective indices ensuring the last frame is included
        # effective_indices = np.linspace(0, segement_length-1, num=(segement_length + sample_rate - 1) // sample_rate, dtype=int)
        effective_indices= np.arange(0, segment_length-1, sample_rate)
        effective_length = len(effective_indices)
        
        cal_len += effective_length
        
        # Fill labels and boundaries
        seg = np.full(effective_length, label)
        bound = np.full(effective_length, 0)  # Default intermediate
    
        if effective_length > 10:
            bound[0:5] = 1  # Start boundary
            bound[-5:] = 2  # End boundary
        elif effective_length <= 10:
            mid_point = effective_length // 2
            bound[0:mid_point] = 1  # Start boundary for first half
            bound[mid_point:] = 2  # End boundary for second half
    
        # bound[0:5]= 1;bound[5: effective_length-5]=0; bound[effective_length-5:effective_length+1]=2;
        labels.extend(seg)
        boundaries.extend(bound)
        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'phase': act,
            'label': actiondict[act]
        })
    labels_arr= np.array(labels)
    bounds_arr= np.array(boundaries)
    return segments,labels_arr,bounds_arr,endf
        
    
def make_full_df(files_df,actiondict, sample_rate):
    num_frames= []
    allsegmetns=[]
    feature_path=[]
    alllabels=[]
    allboundaries=[]
    feature_path=[]
    # vid=[]
    video_ids=[]
    
    
    for i,r in files_df.iterrows():
        annot_file = r['annotation_path']
        video_file = r['video_path']
        vid_name= video_file.split('/')[-1].split('.')[0]

        class_root_path = Path(video_file).parents[1]
        f_path = str(class_root_path) + '/features'
        l_path = str(class_root_path) + '/labels'
        b_path = str(class_root_path) + '/boundaries'
        feat_path = f_path + '/' + vid_name + '.npy'
        label_path= l_path + '/'+str(sample_rate)+'_' + vid_name + '.npy'
        bound_path= b_path + '/'+str(sample_rate)+'_' + vid_name + '.npy'
        feature_path.append(feat_path)
        video_ids.append(vid_name)


        # vid= get_video_prop(video_file)
        # total_frames= vid['num_frames']
        jigsaw_segments, label, bound, num_frame = calculate_jigsaw_segments(annot_file, actiondict,
                                                                             sample_rate=sample_rate)
        np.save(label_path,label)
        np.save(bound_path,bound)


        num_frames.append(num_frame)
        allsegmetns.append(jigsaw_segments)
        alllabels.append(label_path)
        allboundaries.append(bound_path)


    files_df['feature_path'] = feature_path
    files_df['frames'] = num_frames
    files_df['segments'] = allsegmetns
    files_df['labels'] = alllabels
    files_df['boundaries'] = allboundaries
    files_df['video_id'] =video_ids
    full_df = files_df

    #     p = Path(video_file).parents[1]
    #     p= str(p)+'/features'
    #     feat_path= p+'/'+n+'.npy'
    #     if not os.path.exists(feat_path):
    #         print(feat_path)
    #         break

    #     feature_path.append(feat_path)
    #     # vid= get_video_prop(video_file)
    #     # total_frames= vid['num_frames']
    #     jigsaw_segments,label,bound,num_frame = calculate_jigsaw_segments(annot_file,sample_rate=sample_rate,actiondict=actiondict)
    #     total_frames= num_frame
    #     num_frames.append(num_frame)
    #     allsegmetns.append(jigsaw_segments)
    #     alllabels.append(label)
    #     allboundaries.append(bound)
    #     video_ids.append(n)
        
    # files_df['feature_path']=feature_path
    # files_df['frames']=num_frames
    # files_df['segments']=allsegmetns
    # files_df['labels']=alllabels
    # files_df['boundaries']=allboundaries
    # files_df['video_id']= vid
    # full_df= files_df
    return full_df
    