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

#dataloader for HOH_3dm
@DATASETS.register_module()
class HOH_3dm(data.Dataset):
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
            filename_parts = line.split('_')
            taxonomy_id = filename_parts[0]
            model_id = filename_parts[-1].split('.')[0]
            line = f"{taxonomy_id}-{model_id}.npy"
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        
            
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
       

    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        capture_dirs = sample["taxonomy_id"]
        handover_idx = sample["model_id"]
        model_max_scaled = {}
        
        sh = subject_handler.SubjectHandler(subject_data_root_path='$ROOTDIR/subject_data', dyad=capture_dirs[:11])
        # get set number that capture is in
        capture_set = sh.dyad_set
        # print(capture_set)
        obj_id_list = sh.object_list(capture_dirs)
        obj_ID_used = obj_id_list[int(handover_idx)]
        try:
            #using the capture and the handover idx load the data for the HOH dataset
            json_path = r"$ROOTDIR/dev/dataset_extension/capture_reference_files/{}.json".format(capture_dirs)
            with open(json_path, "r") as json_file:
                data = json.load(json_file)
                left_giver = data["left_giver"]
                keyframes = data["keyframes"]
                keyframes_idx = keyframes[int(handover_idx)]
                T_frame_value = keyframes_idx["t_frame"]
                G_frame_value = keyframes_idx["g_frame"]
                O_frame_value = keyframes_idx["o_frame_idx"]
                threedm_used = data["obj"][int(handover_idx)]
            Tpre = T_frame_value - 15
            #get transformations 
            icp_path = f"$ROOTDIR/dev/dataset_extension/full_ptc_object_video/icp_alignments/files/{capture_dirs}_{handover_idx}_transformations.json"
            transforms = json.load(open(icp_path, "r"))
            #G -> O and 3dm->O
            G_to_O = np.linalg.inv(np.array(transforms[f"{O_frame_value}_{G_frame_value}"]))
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
            

            
        
            
            #Access the HOH 3D model from csvs
            #first three columns are points and the last three columns are vertex normals 
           
            csv_file_path = f"$ROOTDIR/3d_models_resampledcsvs/{threedm_used}.csv"
            df = pd.read_csv(csv_file_path,header=None)
            ptcld_points = df.iloc[:, :3].to_numpy()
            normal_points = df.iloc[:, -3:].to_numpy()
            #create a trimesh mesh
            threedm = trimesh.Trimesh(vertices = ptcld_points, vertex_normals = normal_points)
            threedm.apply_transform(threedm_to_O)
            
            

           #Apply different transformations depending on the capture handover of the HOH dataset
            if left_giver:
                if capture_set == 1:
                    
                    left_to_right_transform = [ [-9.99836555e-01, -1.65137177e-02, -7.35935847e-03,  2.12242427e+02],
                                                [-1.65137177e-02,  6.68470043e-01,  7.43555713e-01, -9.92428362e+02],
                                                [-7.35935847e-03,  7.43555713e-01, -6.68633488e-01,  2.23531672e+03],
                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

                    threedm.apply_transform(left_to_right_transform)    
                    set1_to_set2_transform = [  [  1.,     0.,     0.,   41.45],
                                                [  0.,     1.,     0.,     28.08],
                                                [  0.,     0.,     1.,    -16.97],
                                                [  0.,     0.,     0.,     1.  ]]

                    threedm.apply_transform(set1_to_set2_transform)   
                    
                    
                else:
                    
                    
                    left_to_right_transform = [ [-9.99836555e-01, -1.65137177e-02, -7.35935847e-03,  2.12242427e+02],
                                                [-1.65137177e-02,  6.68470043e-01,  7.43555713e-01, -9.92428362e+02],
                                                [-7.35935847e-03,  7.43555713e-01, -6.68633488e-01,  2.23531672e+03],
                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
                    threedm.apply_transform(left_to_right_transform)     
                    set2_fix_transform = [  [  1.,     0.,     0.,   62.76],
                                            [  0.,     1.,     0.,     18.64],
                                            [  0.,     0.,     1.,    -36.50],
                                            [  0.,     0.,     0.,     1.  ]]
                    threedm.apply_transform(set2_fix_transform)       
                 
            
            final_normals = threedm.vertex_normals
            threedm_points = threedm.vertices

            #apply bounding box centering to the 3dm
            bounds_input = np.array([np.min(threedm_points, axis=0), np.max(threedm_points, axis=0)])
            center = np.mean(bounds_input, axis=0)
            model_max_scaled["center_obj"] = center.tolist()
          
            
            translation = -1 * center
            
            threedm_points = threedm_points + translation       
            
            

            
            #get the giver hand at G
            Gh_frame_path = f"$ROOTDIR/{capture_dirs}/PointClouds/filtered/{handover_idx}/Cleaned/giver_frame{G_frame_value}.ply"
            Gh_ptcld = trimesh.load(Gh_frame_path)
            Gh_ptcld.apply_transform(G_to_O)
            #combine object at G and giver hand at G to make the complete data
            output_data = Gh_ptcld.vertices

            ### supersample/ subsample the output_data data##
            if output_data.shape[0] > 1644:
                output_data = self.subsample_trimesh_point_cloud(output_data,1644)
            elif output_data.shape[0] < 1644:
                output_data = self.supersample_vertices(output_data, 1644)
                if output_data.shape[0] > 1644:
                    output_data = self.subsample_trimesh_point_cloud(output_data,1644)
                



           #Apply different transformations depending on the capture handover of the HOH dataset
            if left_giver:
                if capture_set == 1:
                    output_data = trimesh.PointCloud(vertices = output_data)
                    left_to_right_transform = [ [-9.99836555e-01, -1.65137177e-02, -7.35935847e-03,  2.12242427e+02],
                                                [-1.65137177e-02,  6.68470043e-01,  7.43555713e-01, -9.92428362e+02],
                                                [-7.35935847e-03,  7.43555713e-01, -6.68633488e-01,  2.23531672e+03],
                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

                    output_data.apply_transform(left_to_right_transform)
                    set1_to_set2_transform = [  [  1.,     0.,     0.,   41.45],
                                                [  0.,     1.,     0.,     28.08],
                                                [  0.,     0.,     1.,    -16.97],
                                                [  0.,     0.,     0.,     1.  ]]

                    output_data.apply_transform(set1_to_set2_transform)
                    output_data = output_data.vertices
                else:
                    output_data = trimesh.PointCloud(vertices = output_data)
                    left_to_right_transform = [ [-9.99836555e-01, -1.65137177e-02, -7.35935847e-03,  2.12242427e+02],
                                                [-1.65137177e-02,  6.68470043e-01,  7.43555713e-01, -9.92428362e+02],
                                                [-7.35935847e-03,  7.43555713e-01, -6.68633488e-01,  2.23531672e+03],
                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
                    output_data.apply_transform(left_to_right_transform)
                    set2_fix_transform = [  [  1.,     0.,     0.,   62.76],
                                            [  0.,     1.,     0.,     18.64],
                                            [  0.,     0.,     1.,    -36.50],
                                            [  0.,     0.,     0.,     1.  ]]
                    output_data.apply_transform(set2_fix_transform)
                    output_data = output_data.vertices

            
            
            #subsample/supersample the object at G frame 
            #get the Obj at G frame to align the O+G pointcloud
            G_obj_frame_path = f"$ROOTDIR/{capture_dirs}/PointClouds/filtered/{handover_idx}/Cleaned/object_frame{G_frame_value}.ply"
            G_ptclds = trimesh.load(G_obj_frame_path)
            #transform to O
            G_ptclds.apply_transform(G_to_O)
            #subsample
            if G_ptclds.vertices.shape[0] > 1644:
                G_ptclds = self.subsample_trimesh_point_cloud(G_ptclds.vertices,1644)
            elif G_ptclds.vertices.shape[0] < 1644:
                G_ptclds = self.supersample_vertices(G_ptclds.vertices, 1644)
                if G_ptclds.shape[0] > 1644:
                    G_ptclds = self.subsample_trimesh_point_cloud(G_ptclds,1644)
            else:
                G_ptclds = G_ptclds.vertices

            
            #get the bounding box and center
            bounds_output = np.array([np.min(G_ptclds, axis=0), np.max(G_ptclds, axis=0)])
            # Calculate the center and translation vector
            center = np.mean(bounds_output, axis=0)
            model_max_scaled["center_gh"] = center.tolist()
            translation = -1 * center
            
            # Step 3: Apply the centering transformation to the NumPy point cloud
            output_data = output_data + translation 
            
            
            

            #scale the models based on the 3d models
            file_path = r"$ROOTDIR/GraspPC/HOHpc/3d_models.json"
            capture = sample['taxonomy_id']
            idx = sample['model_id']
            handover_idx = int(idx)
            sh = subject_handler.SubjectHandler(subject_data_root_path='$ROOTDIR/subject_data', dyad=capture[:11])
            obj_id_list = sh.object_list(capture)
            object_ID_used = obj_id_list[handover_idx]

            with open(file_path, "r") as json_file:
                json_data = json.load(json_file)

            if object_ID_used in json_data:
                value = json_data[object_ID_used]
                max_value = value["max"]
                output_data = output_data / max_value
                threedm_points = threedm_points / max_value         
                
            
            
            #get the partial (input object) and the gt (giver hand) and the normals to return
            input_data = torch.from_numpy(threedm_points).float()
            data['partial'] = input_data

            
            output_data = torch.from_numpy(output_data).float()
            data['gt'] = output_data
            
            threedm_normal_data = torch.from_numpy(final_normals).float()
            data["normals"] = threedm_normal_data
            
        except Exception as e:
            print(e)
            partial = np.zeros((1644, 3), dtype=np.float32)
            complete = np.zeros((1644, 3), dtype=np.float32)
            normals = np.zeros((1644, 3), dtype=np.float32)
            partial = torch.from_numpy(partial).float()
            complete = torch.from_numpy(complete).float()
            normals = torch.from_numpy(normals).float()
            data['partial'] = partial
            data['gt'] = complete
            data["normals"] = normals


        


    
        return sample['taxonomy_id'], sample['model_id'], obj_ID_used, data['partial'], data['gt'], data["normals"]
    
 

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

    def compare_point_clouds(self, bounds1, bounds2, size_threshold=0.5):
        # Assuming bounds1 and bounds2 are NumPy arrays representing the bounds of point clouds

        # Check if one set of bounds is a lot smaller than the other based on a threshold
        if np.all(bounds1[1] - bounds1[0] < size_threshold * (bounds2[1] - bounds2[0])):
            return True
        elif np.all(bounds2[1] - bounds2[0] < size_threshold * (bounds1[1] - bounds1[0])):
            return True
        else:
            return False
    
    def supersample_trimesh_with_normals(self, tri_mesh, samples_per_vertex, normal_factor=0.01):
        """
        Supersamples points and normals of a trimesh object while preserving normals.

        Parameters:
        - tri_mesh: trimesh.Trimesh
            The input trimesh object.
        - samples_per_vertex: int
            Number of samples to generate per original vertex.
        - normal_factor: float
            Factor to control the extent of normal perturbation.

        Returns:
        - supersampled_points: numpy.ndarray
            Supersampled points of the trimesh.
        - supersampled_normals: numpy.ndarray
            Supersampled normals of the trimesh.
        """
        

        # Get the original points and normals
        original_points = tri_mesh.vertices
        original_normals = tri_mesh.vertex_normals

        # Perturb normals
        perturbed_normals = original_normals + normal_factor * np.random.normal(size=original_normals.shape)

        # Make sure perturbed normals are unit vectors
        perturbed_normals /= np.linalg.norm(perturbed_normals, axis=1)[:, np.newaxis]

        # Initialize arrays for supersampled points and normals
        supersampled_points = []
        supersampled_normals = []

        # Supersample each vertex
        for i in range(len(original_points)):
            vertex = original_points[i]
            normal = perturbed_normals[i]

            # Generate samples around the vertex
            samples = np.random.normal(size=(samples_per_vertex, 3))
            supersampled_vertex = vertex + samples

            # Add supersampled points and normals to the arrays
            supersampled_points.append(supersampled_vertex)
            supersampled_normals.extend([normal] * samples_per_vertex)

        # Concatenate arrays to get the final supersampled points and normals
        supersampled_points = np.concatenate(supersampled_points, axis=0)
        supersampled_normals = np.array(supersampled_normals)

        return supersampled_points, supersampled_normals
    
    def __len__(self):
        return len(self.file_list)