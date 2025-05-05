import argparse
import os
import h5py
import pickle
from ETH_sequence import ETHSequence
import numpy as np

def str2bool(v):
    return v.lower() in ("true", "1")

# Parse command line arguments.
parser = argparse.ArgumentParser(description='Process ETH3D sequences.')
parser.add_argument('--raw_data_path', type=str, default='../../ETHdataset/',
  help='raw data path. default:../../ETHdataset/')
parser.add_argument('--dump_dir', type=str, default='../data_dump/',
  help='data dump path. default:../data_dump/')
parser.add_argument('--desc_name', type=str, default='sift-1000',
  help='prefix of desc filename, default:sift-1000')
parser.add_argument('--vis_th', type=int, default=50,
  help='visibility threshold (not used but kept for compatibility)')
parser.add_argument('--pair_num', type=int, default=1000,
  help='pair num. 1000 for test seq')
parser.add_argument('--sequences', type=str, default='botanical_garden',
  help='ETH3D sequences to process, comma separated')
parser.add_argument('--output_file', type=str, default='ETH-botanical_garden-sift-1000-test.hdf5',
  help='Output HDF5 filename')

class ETHDataset:
    def __init__(self, dataset_path, dump_dir, dump_file, seqs, mode, desc_name, vis_th, pair_num):
        """
        Create an ETH3D dataset from multiple sequences
        
        Args:
            dataset_path: Base path to ETH dataset (e.g., "../../ETHdataset/")
            dump_dir: Directory to save processed data
            dump_file: Output HDF5 filename
            seqs: List of sequence names to process (e.g., ["botanical_garden", "boulders"])
            mode: Mode string (usually "test" for evaluation)
            desc_name: Descriptor name (e.g., "sift-1000")
            vis_th: Visibility threshold (not used for ETH3D)
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
    
    def collect(self):
        """Collect data from all sequences into a single HDF5 file"""
        # Include Rs_true and ts_true in data types as they are now provided by colmap_dataset
        data_types = ['xs', 'ys', 'Rs', 'ts', 'ratios', 'mutuals',
                     'cx1s', 'cy1s', 'f1s', 'cx2s', 'cy2s', 'f2s',
                     'Rs_true', 'ts_true']
        
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
                    pkl_file = os.path.join(seq_dump_dir, f"{tp}.pkl")
                    if os.path.exists(pkl_file):
                        with open(pkl_file, 'rb') as fp:
                            data_seq[tp] = pickle.load(fp)
                    else:
                        print(f"  Warning: {tp}.pkl not found for sequence {seq}")
                        data_seq[tp] = []
                
                # Check if we have data
                if not data_seq.get('xs', []):
                    print(f"  No data found for sequence {seq}")
                    continue
                
                # Get sequence length
                seq_len = len(data_seq['xs'])
                print(f"  Adding {seq_len} pairs from sequence {seq}")
                
                # Add data to HDF5 file
                for i in range(seq_len):
                    # Add all data types
                    for tp in data_types:
                        if tp in data_seq and i < len(data_seq[tp]):
                            data_item = data_seq[tp][i]
                            if tp in ['cx1s', 'cy1s', 'cx2s', 'cy2s', 'f1s', 'f2s'] and not isinstance(data_item, np.ndarray):
                                data_item = np.asarray([data_item])
                            data_i = data[tp].create_dataset(str(pair_idx), data_item.shape, dtype='float32')
                            data_i[:] = data_item
                    
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
            
            # Process sequence using the modified ETHSequence class that uses colmap_dataset
            sequence = ETHSequence(
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
            
            # Save the pairs information for later use
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
    
    # Create ETH dataset
    eth_dataset = ETHDataset(
        dataset_path=config.raw_data_path,
        dump_dir=config.dump_dir,
        dump_file=config.output_file,
        seqs=sequences,
        mode='test',
        desc_name=config.desc_name,
        vis_th=config.vis_th,
        pair_num=config.pair_num
    )