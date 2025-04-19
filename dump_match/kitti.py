import argparse
import os
import h5py
import pickle
from kitti_sequence import KITTISequence
import numpy as np

def str2bool(v):
    return v.lower() in ("true", "1")

# Parse command line arguments.
parser = argparse.ArgumentParser(description='Process KITTI sequences.')
parser.add_argument('--raw_data_path', type=str, default='../../Datasets_filtered/',
  help='raw data path. default:../../Datasets_filtered/')
parser.add_argument('--dump_dir', type=str, default='../data_dump/',
  help='data dump path. default:../data_dump/')
parser.add_argument('--desc_name', type=str, default='sift-1000',
  help='prefix of desc filename, default:sift-1000')
parser.add_argument('--vis_th', type=int, default=50,
  help='visibility threshold (not used for KITTI but kept for compatibility)')
parser.add_argument('--pair_num', type=int, default=1000,
  help='pair num. 1000 for test seq')
parser.add_argument('--sequences', type=str, default='00,01,02,03,04,05,06,07,08,09,10',
  help='KITTI sequences to process, comma separated')
parser.add_argument('--output_file', type=str, default='kitti-08-sift-1000-test.hdf5',
  help='Output HDF5 filename')

class KITTIDataset:
    def __init__(self, dataset_path, dump_dir, dump_file, seqs, mode, desc_name, vis_th, pair_num):
        """
        Create a KITTI dataset from multiple sequences
        
        Args:
            dataset_path: Base path to KITTI dataset (e.g., "../../Datasets_filtered/")
            dump_dir: Directory to save processed data
            dump_file: Output HDF5 filename
            seqs: List of sequence IDs to process (e.g., ["00", "01"])
            mode: Mode string (usually "test" for evaluation)
            desc_name: Descriptor name (e.g., "sift-1000")
            vis_th: Visibility threshold (not used for KITTI)
            pair_num: Number of pairs per sequence
        """
        self.dataset_path = dataset_path
        self.dump_dir = dump_dir
        self.dump_file = os.path.join(dump_dir, dump_file)
        self.seqs = seqs
        self.mode = mode
        self.desc_name = desc_name
        self.vis_th = vis_th
        self.pair_num = pair_num
        
        # Process data
        self.process_data()
    
    def load_poses_rel(self, seq):
        """Load ground truth relative poses for a sequence
        
        Args:
            seq: Sequence ID (e.g., "00")
            
        Returns:
            Dictionary mapping frame indices to (R_true, t_true) tuples
        """
        poses_path = os.path.join(self.dataset_path, "poses_rel", f"{seq}.txt")
        if not os.path.exists(poses_path):
            print(f"Warning: Ground truth poses file not found: {poses_path}")
            return {}
            
        poses = {}
        try:
            with open(poses_path, 'r') as f:
                for i, line in enumerate(f):
                    values = [float(v) for v in line.strip().split()]
                    if len(values) != 12:  # 9 for R + 3 for t
                        print(f"Warning: Invalid line format in {poses_path}, line {i+1}")
                        continue
                        
                    # Extract R (first 9 values) and t (last 3 values)
                    R_true = np.array(values[:9]).reshape(3, 3).astype(np.float32)
                    t_true = np.array(values[9:]).reshape(3, 1).astype(np.float32)
                    
                    poses[i] = (R_true, t_true)
            
            print(f"Loaded {len(poses)} ground truth poses for sequence {seq}")
            return poses
        except Exception as e:
            print(f"Error loading ground truth poses from {poses_path}: {e}")
            return {}
    
    def collect(self):
        """Collect data from all sequences into a single HDF5 file"""
        data_types = ['xs', 'ys', 'Rs', 'ts', 'ratios', 'mutuals',
                     'cx1s', 'cy1s', 'f1s', 'cx2s', 'cy2s', 'f2s',
                     'Rs_true', 'ts_true']  # Added Rs_true and ts_true
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(self.dump_file), exist_ok=True)
        
        # Initialize HDF5 file
        pair_idx = 0
        with h5py.File(self.dump_file, 'w') as f:
            # Create groups for each data type
            data = {}
            for tp in data_types:
                data[tp] = f.create_group(tp)
                
            # Process each sequence
            for seq in self.seqs:
                print(f"Collecting data from sequence {seq}")
                
                # Check if sequence has processed data
                seq_dump_dir = os.path.join(self.dump_dir, seq, self.desc_name, self.mode)
                ready_file = os.path.join(seq_dump_dir, "ready")
                
                if not os.path.exists(ready_file):
                    print(f"  Sequence {seq} not processed, skipping")
                    continue
                
                # Load sequence data
                data_seq = {}
                for tp in data_types:
                    if tp in ['Rs_true', 'ts_true']:
                        continue  # These are loaded separately
                        
                    pkl_file = os.path.join(seq_dump_dir, f"{tp}.pkl")
                    if os.path.exists(pkl_file):
                        with open(pkl_file, 'rb') as fp:
                            data_seq[tp] = pickle.load(fp)
                    else:
                        print(f"  Warning: {tp}.pkl not found for sequence {seq}")
                        data_seq[tp] = []
                
                # Load ground truth poses data
                gt_poses = self.load_poses_rel(seq)
                
                # Load image pairs information to match with ground truth poses
                pairs_file = os.path.join(seq_dump_dir, "pairs.pkl")
                pairs = []
                if os.path.exists(pairs_file):
                    with open(pairs_file, 'rb') as fp:
                        pairs = pickle.load(fp)
                        
                # Check if we have data
                if not data_seq.get('xs', []):
                    print(f"  No data found for sequence {seq}")
                    continue
                
                # Get sequence length
                seq_len = len(data_seq['xs'])
                print(f"  Adding {seq_len} pairs from sequence {seq}")
                
                # Add data to HDF5 file
                for i in range(seq_len):
                    # Add standard data
                    for tp in data_types:
                        if tp in ['Rs_true', 'ts_true']:
                            continue  # Handle these separately
                            
                        if tp in data_seq and i < len(data_seq[tp]):
                            data_item = data_seq[tp][i]
                            if tp in ['cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s']:
                                data_item = np.asarray([data_item]) if not isinstance(data_item, np.ndarray) else data_item
                            data_i = data[tp].create_dataset(str(pair_idx), data_item.shape, dtype='float32')
                            data_i[:] = data_item
                    
                    # Add ground truth poses if available
                    if i < len(pairs) and pairs[i][0] in gt_poses:
                        frame_idx = pairs[i][0]
                        R_true, t_true = gt_poses[frame_idx]
                        
                        # Create datasets for ground truth R and t
                        data_R_true = data['Rs_true'].create_dataset(str(pair_idx), R_true.shape, dtype='float32')
                        data_R_true[:] = R_true
                        
                        data_t_true = data['ts_true'].create_dataset(str(pair_idx), t_true.shape, dtype='float32')
                        data_t_true[:] = t_true
                    
                    pair_idx += 1
                
                print(f"  Pair index now: {pair_idx}")
    
    def process_data(self):
        """Process all sequences"""
        # Make sure dump directory exists
        os.makedirs(self.dump_dir, exist_ok=True)
        
        # Process each sequence
        for seq in self.seqs:
            print(f"\nProcessing sequence {seq}")
            
            # Check if sequence directory exists
            seq_path = os.path.join(self.dataset_path, seq)
            if not os.path.exists(seq_path):
                print(f"Sequence path does not exist: {seq_path}")
                continue
                
            # Create output directory
            seq_dump_dir = os.path.join(self.dump_dir, seq, self.desc_name, self.mode)
            os.makedirs(seq_dump_dir, exist_ok=True)
            
            # Process sequence
            sequence = KITTISequence(
                dataset_path=seq_path,
                dump_dir=seq_dump_dir,
                desc_name=self.desc_name,
                vis_th=self.vis_th,
                pair_num=self.pair_num
            )
            
            print('Computing nearest neighbors...')
            sequence.dump_intermediate()
            
            print('Processing image pairs...')
            sequence.dump_datasets()
            
            # Save the pairs information for later use in matching with ground truth poses
            with open(os.path.join(seq_dump_dir, "pairs.pkl"), 'wb') as f:
                pickle.dump(sequence.pairs, f)
        
        # Collect data from all sequences
        print('\nCollecting data from all sequences...')
        self.collect()
        print(f"Final dataset saved to {self.dump_file}")


if __name__ == "__main__":
    config = parser.parse_args()
    
    # Process sequence IDs from comma-separated list
    sequences = config.sequences.split(',')
    print(f'Processing {len(sequences)} sequences: {sequences}')
    
    # Create KITTI dataset
    kitti_dataset = KITTIDataset(
        dataset_path=config.raw_data_path,
        dump_dir=config.dump_dir,
        dump_file=config.output_file,
        seqs=sequences,
        mode='test',
        desc_name=config.desc_name,
        vis_th=config.vis_th,
        pair_num=config.pair_num
    )