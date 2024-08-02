import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
import logging
import open3d as o3d
import trimesh
import random
import json
import csv
import pandas as pd
import sys
sys.path.insert(0, os.path.join('$ROOTDIR/''python'))
import handler_subject_data as subject_handler
import handler as capture_handler
import constants
import utils as handover_utils
import pointcloud_creation
import utils_3d
import handler_calib as calib_handler

#dataloader for DexYCB
@DATASETS.register_module()
class DexYCB(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
       
        print(self.data_list_file)
        

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            self.file_list.append({ 
                'file_path': line
            })
        
            
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
       

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        model_max_scaled = {}
        
       
        name = sample["file_path"].split(".")[0]
        
        try:
            
            

            #load in the object point cloud input 
            dexycb_input_path = f"$ROOTDIR/dexycb/dexycb_csv/{name}_object.csv"
            df = pd.read_csv(dexycb_input_path,header=None)
            dexycb_object_points = df.iloc[:, :3].to_numpy()
            dexycb_normal_points = df.iloc[:, -3:].to_numpy()
            
            

            #load in the hand point cloud 
            dexycb_output_path = f"$ROOTDIR/dexycb/dexycb_csv/{name}_hand.csv"
            df = pd.read_csv(dexycb_output_path,header=None)
            dexycb_hand_ptcld_points = df.iloc[:, :3].to_numpy()
           

            #get the points and normals to return
            input_object_data = torch.from_numpy(dexycb_object_points).float()
            input_object_normals = torch.from_numpy(dexycb_normal_points).float()
            output_hand_data = torch.from_numpy(dexycb_hand_ptcld_points).float()



            data['object_points'] = input_object_data
            data['object_normals'] = input_object_normals
            data['hand_points'] = output_hand_data

            
                
        except Exception as e:
            print(e)
            name = name
            input_object_data =  np.zeros((1644, 3), dtype=np.float32)
            input_object_normals =  np.zeros((1644, 3), dtype=np.float32)
            output_hand_data  =  np.zeros((1644, 3), dtype=np.float32)
            data['object_points'] = input_object_data
            data['object_normals'] = input_object_normals
            data['hand_points'] = output_hand_data

        return name,  data['object_points'], data["object_normals"], data['hand_points']
        
    
    def __len__(self):
        return len(self.file_list)