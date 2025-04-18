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

class KITTISequence(object):
    def __init__(self, dataset_path, dump_dir, desc_name, vis_th, pair_num, pair_name=None):
        """
        Initialize a KITTI sequence for processing
        
        Args:
            dataset_path: Path to KITTI sequence directory (e.g. "../../Datasets_filtered/00/")
            dump_dir: Directory to save processed data
            desc_name: Descriptor name suffix (e.g. "sift-1000")
            vis_th: Not used for KITTI, kept for compatibility
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
        
        # Load calibration data
        self.calib = self.load_calib(os.path.join(self.data_path, "calib.txt"))
        
        # Get image paths
        self.image_fullpath_list = sorted(glob.glob(os.path.join(self.data_path, "image_0", "*.png")))
        print(f"Found {len(self.image_fullpath_list)} images")
        
        # Create image pairs - sequential pairs for KITTI
        if pair_name is None:
            # Generate pairs from consecutive frames
            self.pairs = []
            for i in range(len(self.image_fullpath_list) - 1):
                self.pairs.append((i, i + 1))
            
            # Randomly select pairs if we have too many
            if len(self.pairs) > pair_num:
                np.random.seed(1234)
                self.pairs = [self.pairs[i] for i in np.random.permutation(len(self.pairs))[:pair_num]]
        else:
            with open(pair_name, 'rb') as f:
                self.pairs = pickle.load(f)
                
        print(f'Created {len(self.pairs)} image pairs')
        
        # Try to load ground truth poses if available
        self.poses_rel = self.load_ground_truth_poses()

    def load_ground_truth_poses(self):
        """Load ground truth relative poses from poses_rel directory"""
        poses_dir = os.path.join(os.path.dirname(os.path.dirname(self.data_path)), "poses_rel")
        poses_file = os.path.join(poses_dir, f"{self.seq_id}.txt")
        
        poses = {}
        if not os.path.exists(poses_file):
            print(f"Warning: Ground truth poses file not found: {poses_file}")
            return poses
            
        try:
            with open(poses_file, 'r') as f:
                for i, line in enumerate(f):
                    values = [float(v) for v in line.strip().split()]
                    if len(values) == 12:  # 9 for R + 3 for t
                        # Extract R (first 9 values) and t (last 3 values)
                        R_true = np.array(values[:9]).reshape(3, 3).astype(np.float32)
                        t_true = np.array(values[9:]).reshape(3, 1).astype(np.float32)
                        poses[i] = (R_true, t_true)
            
            print(f"Loaded {len(poses)} ground truth poses for sequence {self.seq_id}")
        except Exception as e:
            print(f"Error loading ground truth poses: {e}")
        
        return poses

    def load_calib(self, calib_file):
        """Load KITTI calibration file"""
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            
        calib = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key.strip()] = np.array([float(x) for x in value.split()])
        
        # Create camera matrix from P0 (left grayscale camera projection matrix)
        P0 = calib['P0'].reshape(3, 4)
        K = P0[:3, :3]  # Intrinsic matrix is the first 3x3 part of P
        
        # Return data in a format similar to what sequence.py expects
        return {
            'K': K, 
            'img_size': np.array([1241, 376])  # Standard KITTI image size
        }

    def dump_nn(self, ii, jj):
        """Compute and save nearest neighbor matches"""
        dump_file = os.path.join(self.intermediate_dir, f"nn-{ii}-{jj}.h5")
        if not os.path.exists(dump_file):
            image_i, image_j = self.image_fullpath_list[ii], self.image_fullpath_list[jj]
            # Construct the feature file paths
            feature_i = os.path.join(self.data_path, "features", f"{os.path.basename(image_i)}.{self.desc_name}.hdf5")
            feature_j = os.path.join(self.data_path, "features", f"{os.path.basename(image_j)}.{self.desc_name}.hdf5")
            
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
        w, h = 1241, 376  # Standard KITTI image size
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
        image_i, image_j = self.image_fullpath_list[ii], self.image_fullpath_list[jj]
        
        # Construct feature paths
        feature_i = os.path.join(self.data_path, "features", f"{os.path.basename(image_i)}.{self.desc_name}.hdf5")
        feature_j = os.path.join(self.data_path, "features", f"{os.path.basename(image_j)}.{self.desc_name}.hdf5")
        
        # Check feature files
        if not os.path.exists(feature_i) or not os.path.exists(feature_j):
            print(f"Feature files missing for pair {ii}, {jj}")
            return []
            
        # Check NN file
        nn_file = os.path.join(self.intermediate_dir, f"nn-{ii}-{jj}.h5")
        if not os.path.exists(nn_file):
            print(f"NN file missing for pair {ii}, {jj}")
            return []
        
        # Get camera intrinsics (same for both images in KITTI)
        K = self.calib['K']
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
            
            # For KITTI we'll estimate motion between consecutive frames using 8-point algorithm
            # This is simplistic but matches the YFCC approach conceptually
            # In practice, a more sophisticated method might be used for actual training data
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
                
                # Get ground truth R and t from poses_rel if available
                R_true = None
                t_true = None
                if ii in self.poses_rel:
                    R_true, t_true = self.poses_rel[ii]
                
                # Return tuple including R_true and t_true if available
                if R_true is not None and t_true is not None:
                    return xs, ys, R, t, ratio_test, mutual_nearest, cx, cy, f[0], cx, cy, f[0], R_true, t_true
                else:
                    return xs, ys, R, t, ratio_test, mutual_nearest, cx, cy, f[0], cx, cy, f[0]
        except Exception as e:
            print(f"Error processing pair {ii}, {jj}: {e}")
        
        return []

    def dump_datasets(self):
        """Create and save training data for all pairs"""
        ready_file = os.path.join(self.dump_dir, "ready")
        var_name = ['xs', 'ys', 'Rs', 'ts', 'ratios', 'mutuals', 'cx1s', 'cy1s', 'f1s', 'cx2s', 'cy2s', 'f2s']
        # Add ground truth R and t variables
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
                    
                    # Copy ground truth data if available
                    if len(res) > len(var_name):  # Ground truth data is available
                        res_dict['Rs_true'] += [res[len(var_name)]]
                        res_dict['ts_true'] += [res[len(var_name) + 1]]
                    # If ground truth not available for this pair but we want to keep consistent size
                    elif success_count > 0 and 'Rs_true' in res_dict and len(res_dict['Rs_true']) > 0:
                        # Add placeholder values of same shape as previous entries
                        prev_R_shape = res_dict['Rs_true'][0].shape if res_dict['Rs_true'] else (3, 3)
                        prev_t_shape = res_dict['ts_true'][0].shape if res_dict['ts_true'] else (3, 1)
                        res_dict['Rs_true'] += [np.zeros(prev_R_shape, dtype=np.float32)]
                        res_dict['ts_true'] += [np.zeros(prev_t_shape, dtype=np.float32)]
                        
                    success_count += 1
            
            print(f"\nSuccessfully processed {success_count}/{len(self.pairs)} pairs")
            
            if success_count > 0:
                # Save standard variables
                for name in var_name:
                    if res_dict[name]:  # Only save if there's data
                        out_file_name = os.path.join(self.dump_dir, name) + ".pkl"
                        with open(out_file_name, "wb") as ofp:
                            pickle.dump(res_dict[name], ofp)
                
                # Save ground truth variables if available
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