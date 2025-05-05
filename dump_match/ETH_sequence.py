from __future__ import print_function
import numpy as np
import sys
from tqdm import tqdm
import os
import pickle
import cv2
import itertools
from six.moves import xrange
from feature_match import computeNN
from utils import saveh5, loadh5
from geom import get_episym
import glob
import sys

# Import the colmap_dataset module
sys.path.append("../../")  # Add parent directory to path to import from Datasets
from Datasets.colmap_dataset import ETH3D

class ETHSequence(object):
    def __init__(self, dataset_path, dump_dir, desc_name, vis_th, pair_num, pair_name=None):
        """
        Initialize a ETH sequence for processing
        
        Args:
            dataset_path: Path to ETH sequence directory (e.g. "../../ETHdataset/botanical_garden/")
            dump_dir: Directory to save processed data
            desc_name: Descriptor name suffix (e.g. "sift-1000")
            vis_th: Not used for ETH, kept for compatibility
            pair_num: Number of pairs to process
            pair_name: Pre-defined pairs file path (if None, pairs will be generated)
        """
        self.data_path = dataset_path.rstrip("/") + "/"
        self.dump_dir = dump_dir
        self.desc_name = desc_name
        print('dump dir ' + self.dump_dir)
        
        # Get the sequence ID from the data path
        self.seq_id = os.path.basename(os.path.normpath(self.data_path))
        print(f"Processing sequence: {self.seq_id}")
        
        # Create output directories
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)
        self.intermediate_dir = os.path.join(self.dump_dir, 'dump')
        if not os.path.exists(self.intermediate_dir):
            os.makedirs(self.intermediate_dir)
            
        # Use ETH3D from colmap_dataset to get the data
        print(f"Loading ETH3D data from {self.data_path}")
        self.eth3d = ETH3D(
            path=self.data_path,
            num_points=-1,
            threshold=1,
            max_F=pair_num,
            random=False,
            min_matches=20,
            compute_virtual_points=False,
            mode="test"
        )
        
        # Create pairs from ETH3D data
        self.pairs = []
        self.pts_by_pair = {}
        self.img_paths = []
        self.R_true = []
        self.t_true = []
        self.K = []
        
        # Extract image pairs and data from ETH3D
        for i in range(len(self.eth3d)):
            pts, side_info, R_gt, t_gt, K = self.eth3d[i]
            img1_path, img2_path = self.eth3d.img_paths[i]
            
            # Extract the image filenames without extension as IDs
            img1_name = os.path.basename(img1_path)
            img2_name = os.path.basename(img2_path)
            img1_id = os.path.splitext(img1_name)[0]  # Use filename without extension as ID
            img2_id = os.path.splitext(img2_name)[0]  # Use filename without extension as ID
            
            # Store the pair and related data
            self.pairs.append((img1_id, img2_id))
            
            self.img_paths.append((img1_path, img2_path))
            self.R_true.append(R_gt)
            self.t_true.append(t_gt)
            self.K.append(K)
            
        print(f"Created {len(self.pairs)} image pairs")
        
        # Load feature paths or use paths from the dataset if available
        self.feature_path = os.path.join(self.data_path, "features")
        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)

    def dump_nn(self, ii, jj):
        """Compute and save nearest neighbor matches"""
        dump_file = os.path.join(self.intermediate_dir, f"nn-{ii}-{jj}.h5")
        if not os.path.exists(dump_file):
            image_i, image_j = self.img_paths[self.pairs.index((ii, jj))]
            
            # Extract base filenames without extension
            img1_name = os.path.splitext(os.path.basename(image_i))[0]
            img2_name = os.path.splitext(os.path.basename(image_j))[0]
            
            # Create feature paths with the pattern
            feature_i = os.path.join(self.feature_path, f"{img1_name}.{self.desc_name}.hdf5")
            feature_j = os.path.join(self.feature_path, f"{img2_name}.{self.desc_name}.hdf5")
            
            if not os.path.exists(feature_i) or not os.path.exists(feature_j):
                print(f"Warning: Feature file not found for {feature_i} or {feature_j}")
                return False
                
            try:
                desc_ii = loadh5(feature_i)["descriptors"]
                desc_jj = loadh5(feature_j)["descriptors"]
                
                idx_sort, ratio_test, mutual_nearest = computeNN(desc_ii, desc_jj)
                
                # Dump to disk
                dump_dict = {}
                dump_dict["idx_sort"] = idx_sort
                dump_dict["ratio_test"] = ratio_test
                dump_dict["mutual_nearest"] = mutual_nearest
                saveh5(dump_dict, dump_file)
                return True
            except Exception as e:
                print(f"Error processing features: {e}")
                return False
        return True

    def dump_intermediate(self):
        """Compute and save nearest neighbor matches for all pairs"""
        success_count = 0
        for ii, jj in tqdm(self.pairs):
            if self.dump_nn(ii, jj):
                success_count += 1
        print(f'Done computing nearest neighbors. Successful: {success_count}/{len(self.pairs)}')
        return success_count

    def unpack_K(self, K):
        """Extract camera parameters from calibration matrix"""
        w, h = 1241, 376  # Standard ETH image size
        cx = K[0, 2]
        cy = K[1, 2]
        # Get focals
        fx = K[0, 0]
        fy = K[1, 1]
        return cx, cy, [fx, fy]

    def norm_kp(self, cx, cy, fx, fy, kp):
        """Normalize keypoints"""
        kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
        return kp

    def make_xy(self, ii, jj):
        """Create training data for a pair of images"""
        # Get image paths
        image_i, image_j = self.img_paths[self.pairs.index((ii, jj))]
        
        # Construct feature paths using the new pattern (image_name.sift-1000.hdf5)
        # Extract base filenames without extension
        img1_name = os.path.splitext(os.path.basename(image_i))[0]
        img2_name = os.path.splitext(os.path.basename(image_j))[0]
        
        # Create feature paths with the pattern
        feature_i = os.path.join(self.feature_path, f"{img1_name}.{self.desc_name}.hdf5")
        feature_j = os.path.join(self.feature_path, f"{img2_name}.{self.desc_name}.hdf5")
        
        # Check feature files
        if not os.path.exists(feature_i) or not os.path.exists(feature_j):
            print(f"Feature files missing for pair {ii}, {jj}")
            print(f"Tried to find: {feature_i} and {feature_j}")
            return []
            
        # Check NN file
        nn_file = os.path.join(self.intermediate_dir, f"nn-{ii}-{jj}.h5")
        if not os.path.exists(nn_file):
            print(f"NN file missing for pair {ii}, {jj}")
            return []
        
        # Get camera intrinsics
        K = self.K[self.pairs.index((ii, jj))]
        cx, cy, f = self.unpack_K(K)
        
        # Load keypoints
        try:
            kp_i = loadh5(feature_i)["keypoints"][:, :2]
            kp_j = loadh5(feature_j)["keypoints"][:, :2]
            
            # Load NN matches
            nn_info = loadh5(nn_file)
            idx_sort, ratio_test, mutual_nearest = nn_info["idx_sort"], nn_info["ratio_test"], nn_info["mutual_nearest"]
            
            # Get matched points
            x1 = self.norm_kp(cx, cy, f[0], f[1], kp_i)
            x2 = self.norm_kp(cx, cy, f[0], f[1], kp_j[idx_sort[1], :])
            
            # For ETH we'll estimate motion between consecutive frames using 8-point algorithm
            if len(x1) >= 8:  # Need at least 8 points for 8-point algorithm
                # Normalize coordinates
                x1_norm = x1.copy()
                x2_norm = x2.copy()
                
                # Use OpenCV to estimate Essential matrix
                E, inliers = cv2.findEssentialMat(
                    x1_norm, x2_norm, 
                    np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
                    method=cv2.RANSAC, 
                    prob=0.999, 
                    threshold=0.001
                )
                
                # Recover pose from Essential matrix
                _, R, t, _ = cv2.recoverPose(E, x1_norm, x2_norm)
                
                # Calculate inlier mask based on epipolar constraint
                geod_d = get_episym(x1, x2, R, t)
                ys = geod_d.reshape(-1, 1)
                
                # Format x1 and x2 as in YFCC
                xs = np.concatenate([x1, x2], axis=1).reshape(1, -1, 4)
                
                # Get ground truth R and t from ETH3D data
                R_true = self.R_true[self.pairs.index((ii, jj))]
                t_true = self.t_true[self.pairs.index((ii, jj))]
                
                return xs, ys, R, t, ratio_test, mutual_nearest, cx, cy, f[0], cx, cy, f[0], R_true, t_true
        except Exception as e:
            print(f"Error processing pair {ii}, {jj}: {e}")
        
        return []

    def dump_datasets(self):
        """Create and save training data for all pairs"""
        ready_file = os.path.join(self.dump_dir, "ready")
        var_name = ['xs', 'ys', 'Rs', 'ts', 'ratios', 'mutuals', 'cx1s', 'cy1s', 'f1s', 'cx2s', 'cy2s', 'f2s']
        var_name_gt = ['Rs_true', 'ts_true']
        
        res_dict = {}
        for name in var_name + var_name_gt:
            res_dict[name] = []
            
        if not os.path.exists(ready_file):
            print("\n -- No ready file {}".format(ready_file))
            success_count = 0
            for pair_idx, pair in enumerate(self.pairs):
                print("\rWorking on {} / {}".format(pair_idx, len(self.pairs)), end="")
                sys.stdout.flush()
                res = self.make_xy(pair[0], pair[1])
                if len(res) != 0:
                    # Copy standard data
                    for var_idx, name in enumerate(var_name):
                        if var_idx < len(res):
                            res_dict[name] += [res[var_idx]]
                    
                    # Copy ground truth data
                    res_dict['Rs_true'] += [res[len(var_name)]]
                    res_dict['ts_true'] += [res[len(var_name) + 1]]
                    
                    success_count += 1
            
            print(f"\nSuccessfully processed {success_count}/{len(self.pairs)} pairs")
            
            if success_count > 0:
                # Save standard variables
                for name in var_name:
                    if res_dict[name]:  # Only save if there's data
                        out_file_name = os.path.join(self.dump_dir, name) + ".pkl"
                        with open(out_file_name, "wb") as ofp:
                            pickle.dump(res_dict[name], ofp)
                
                # Save ground truth variables
                for name in var_name_gt:
                    if res_dict[name]:  # Only save if there's data
                        out_file_name = os.path.join(self.dump_dir, name) + ".pkl"
                        with open(out_file_name, "wb") as ofp:
                            pickle.dump(res_dict[name], ofp)
                        
                # Mark ready
                with open(ready_file, "w") as ofp:
                    ofp.write("This folder is ready\n")
            else:
                print("No successful pairs processed, data not saved")
        else:
            print('Dataset already exists!')