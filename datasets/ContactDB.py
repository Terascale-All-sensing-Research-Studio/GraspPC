import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
import logging
import open3d as o3d
import pandas as pd
import trimesh
import csv
import json
import random
import sys
sys.path.insert(0, os.path.join('$ROOTDIR/''python'))
import handler_subject_data as subject_handler
import handler as capture_handler
import constants
import utils as handover_utils
import pointcloud_creation
import utils_3d
import handler_calib as calib_handler

#ContactDB dataloader

@DATASETS.register_module()
class ContactDB(data.Dataset):
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
        
        contactdb_file_path = sample["file_path"]
        filename_contactdb = contactdb_file_path.split(".")[0]
        number_used = contactdb_file_path.split("_")[4]
      
       
        #Since the dataset only has 50 objects, we can use the HOH dataset to get different transformations and use them to randomize the ContactDB objects so that there are more objects in the test set
        threedm_cap_handover_path = "$ROOTDIR/GraspPC/100_random_threedm_transformations.json"
        with open(threedm_cap_handover_path, "r") as json_file:
            data_dict  = json.load(json_file)
        all_keys = list(data_dict.keys())
        obj_ID_used = random.choice(all_keys)
        

        # Randomly select one key
        capture_handover = data_dict[obj_ID_used]
        capture_dirs = capture_handover.split("_")[0]
        handover_idx = capture_handover.split("_")[1]
        
    
        
        
        


    

    
        try:

            #Get the attributes that were given to the ContactDB object from the HOH dataset
            json_path = r"$ROOTDIR/dataset_extension/capture_reference_files/{}.json".format(capture_dirs)
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
                left_giver = data["left_giver"]
            

            #using the random capture and handoveridx get the threedm_to_O transformation used
            #get transformations 
            icp_path = f"$ROOTDIR/dataset_extension/full_ptc_object_video/icp_alignments/files/{capture_dirs}_{handover_idx}_transformations.json"
            transforms = json.load(open(icp_path, "r"))
            #3dm->O
            threedm_to_O = np.array(transforms[f"3dm_to_O"])
            
            
            
            sh = subject_handler.SubjectHandler(dyad=capture_dirs[:11])
            capture_set = sh.dyad_set
            # print(capture_set)
                
            # create new CalibHandler object
            chandler = calib_handler.CalibHandler()
            calib_params_dict = calib_handler.get_corner_params_all_cams(
                chandler,
                capture_set,
                modality="kcolor",
            )
                # get extrinsics
            calib_params_ex = {}
            for corner in ["12","23","03"]:
                params = chandler.construct_Para(modality="kd", calib_set=capture_set,cam_idx=corner)
                calib_params_ex[f"{corner}_rot"] = params.rotation
                calib_params_ex[f"{corner}_tran"] = params.translation
            

            
    
            
            
            #get the ptclds for contactdb 
            contactdb_path = f"$ROOTDIR/contact_db/contactdb_ply/{filename_contactdb}.ply"
            #load in the ptcld
            contactdb_ptcld = trimesh.load(contactdb_path)

            #apply the transformatiosn to the contactdb objects
            rotational_component = threedm_to_O[:3, :3]
            new_transform = np.eye(4)
            new_transform[:3, :3] = rotational_component
            new_transform[:3, 3] = 0 
            contactdb_ptcld.apply_transform(new_transform)
            
        
       
            #based on the capture and handover idx from the HOH dataset, find out which transformations the contactdb object will use
            if left_giver:
                if capture_set == 1:
                    
                    left_to_right_transform = [ [-9.99836555e-01, -1.65137177e-02, -7.35935847e-03,  2.12242427e+02],
                                                [-1.65137177e-02,  6.68470043e-01,  7.43555713e-01, -9.92428362e+02],
                                                [-7.35935847e-03,  7.43555713e-01, -6.68633488e-01,  2.23531672e+03],
                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

                    left_to_right_transform = np.array(left_to_right_transform)
                    rotation_left_to_right = left_to_right_transform[:3,:3]
                    new_rotational_transform = np.eye(4)
                    new_rotational_transform[:3, :3] = rotation_left_to_right
                    new_rotational_transform[:3, 3] = 0 
                    contactdb_ptcld.apply_transform(new_rotational_transform)
        
                else:
                
                    left_to_right_transform = [ [-9.99836555e-01, -1.65137177e-02, -7.35935847e-03,  2.12242427e+02],
                                                [-1.65137177e-02,  6.68470043e-01,  7.43555713e-01, -9.92428362e+02],
                                                [-7.35935847e-03,  7.43555713e-01, -6.68633488e-01,  2.23531672e+03],
                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
                    left_to_right_transform = np.array(left_to_right_transform)
                    rotation_left_to_right = left_to_right_transform[:3,:3]
                    new_rotational_transform = np.eye(4)
                    new_rotational_transform[:3, :3] = rotation_left_to_right
                    new_rotational_transform[:3, 3] = 0 
                    contactdb_ptcld.apply_transform(new_rotational_transform)
            
            
            
            #add the contactdb points to data[points]
            contactdb_points = contactdb_ptcld.vertices
            input_data = torch.from_numpy(contactdb_points).float()
            data['points'] = input_data

            
                
        except Exception as e:
            print(e)
            capture_dirs = capture_dirs
            handover_idx = handover_idx
            obj_ID_used = obj_ID_used
            points = np.zeros((1644, 3), dtype=np.float32)
            normals = torch.from_numpy(points).float()
            data['points'] = points
        

        return filename_contactdb, data['points'], number_used
        


    
    
 

    def assign_camera_index(self, file_list):
        if file_list in self.camera_mapping:
            used_cameras = self.camera_mapping[file_list]
            available_cameras = list(set(range(4)) - set(used_cameras))
            if not available_cameras:
                available_cameras = list(range(4))  # Reset if all cameras were used
            camera_idx = random.choice(available_cameras)
            used_cameras.append(camera_idx)
        else:
            camera_idx = random.randint(0, 3)
            self.camera_mapping[file_list] = [camera_idx]
        return camera_idx
    def subsample_trimesh_point_cloud(self, ptcld, target_num_vertices):
        # Get the vertices and colors of the trimesh
        vertices = np.array(ptcld)
        # mask_vertices = np.array(mask)

        # Ensure the target number of vertices is within bounds
        target_num_vertices = max(min(target_num_vertices, len(vertices)), 1)

        # Calculate the subsampling rate based on the target number of vertices
        subsampling_rate = target_num_vertices / len(vertices)

        # Randomly select the vertices for subsampling
        random_indices = np.random.choice(len(vertices), target_num_vertices, replace=False)

        # Create the subsampled point cloud
        subsampled_vertices = vertices[random_indices]
        # subsampled_mask = mask_vertices[random_indices]

        # return subsampled_vertices, subsampled_mask
        return subsampled_vertices

    def supersample_vertices(self, vertices, target_num_vertices):
        if len(vertices) < 2:
            raise ValueError("At least two vertices are required.")
        if target_num_vertices <= len(vertices): # then we already have too many verts
            return np.array(vertices)
        original_vertices = np.array(vertices)
        total_distance = np.sum(np.linalg.norm(original_vertices[1:] - original_vertices[:-1], axis=1))
        average_segment_length = total_distance / (len(vertices) - 1)
        
        supersampled_vertices = [original_vertices[0]]  # Start with the first vertex
        
        for i in range(1, len(vertices)):
            start_vertex = original_vertices[i - 1]
            end_vertex = original_vertices[i]
            
            direction = end_vertex - start_vertex
            segment_length = np.linalg.norm(direction)
            num_points = int(np.ceil(segment_length / average_segment_length))
            
            interpolated_points = [start_vertex + j * (direction / (num_points + 1))
                                for j in range(1, num_points + 1)]
            
            supersampled_vertices.extend(interpolated_points)
        
        while len(supersampled_vertices) < target_num_vertices:
            remaining = target_num_vertices - len(supersampled_vertices)
            segment_to_duplicate = min(len(vertices) - 1, remaining)
            for i in range(segment_to_duplicate):
                supersampled_vertices.append(original_vertices[i + 1])

        return np.array(supersampled_vertices)
    
    def normalize_pc(self, points, trajectory):

        max_cord = np.max(np.max(points))
        points /= max_cord
        centroid = np.mean(points, axis=0)
        points -= centroid
        # furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        trajectory /= max_cord
        trajectory -= centroid
       
        return points, trajectory
    def combine_point_clouds(self, pc1, pc2):
            # pc1 should be giver ptcld, pc2 should be receiver
            combined_vertices = np.vstack((pc1, pc2))
            # combined_colors = np.vstack((pc1.colors, pc2.colors))

            # combined_pc = trimesh.PointCloud(vertices=combined_vertices)

            # mask = np.concatenate((np.zeros(len(pc1)), np.ones(len(pc2))))

            # return combined_vertices, mask
            return combined_vertices
    def subsample_trimesh(self, tri_mesh, num_points):
        # 
        # Subsamples points and normals of a trimesh object.

        # Parameters:
        # - tri_mesh: trimesh.Trimesh
        #     The input trimesh object.
        # - num_points: int
        #     The number of points to subsample.

        # Returns:
        # - subsampled_points: numpy.ndarray
        #     Subsampled points of the trimesh.
        # - subsampled_normals: numpy.ndarray
        #     Subsampled normals of the trimesh.
        # 
        # Ensure the input is a trimesh object
        if not isinstance(tri_mesh, trimesh.Trimesh):
            raise ValueError("Input must be a trimesh.Trimesh object")

        # Get the points and normals
        points = tri_mesh.vertices
        normals = tri_mesh.vertex_normals

        # Check if the number of points to subsample is valid
        if num_points <= 0 or num_points > len(points):
            raise ValueError("Invalid number of points to subsample")

        # Randomly subsample points
        indices = np.random.choice(len(points), num_points, replace=False)
        subsampled_points = points[indices]
        subsampled_normals = normals[indices]

        return subsampled_points, subsampled_normals

    
   
    def __len__(self):
        return len(self.file_list)