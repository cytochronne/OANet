import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import cv2
import h5py


def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, default='../../Datasets_filtered',
  help='KITTI dataset directory (default: ../../Datasets_filtered).')
parser.add_argument('--img_glob', type=str, default='*/image_0/*.png',
  help='Glob match if directory of images is specified (default: \'*/image_0/*.png\').')
parser.add_argument('--sequences', type=str, default='00,01,02,03,04,05,06,07,08,09,10',
  help='KITTI sequence numbers, comma separated (default: 00,01,02,03,04,05,06,07,08,09,10).')
parser.add_argument('--num_kp', type=int, default='1000',
  help='keypoint number, default:1000')
parser.add_argument('--suffix', type=str, default='sift-1000',
  help='suffix of filename, default:sift-1000')

 

class ExtractSIFT(object):
  def __init__(self, num_kp, contrastThreshold=1e-5):
    self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

  def run(self, img_path):
    img = cv2.imread(img_path)
    # surf, two-direction optical flow
    cv_kp, desc = self.sift.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc

def write_feature(pts, desc, filename):
  with h5py.File(filename, "w") as ifp:
      ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
      ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
      ifp["keypoints"][:] = pts
      ifp["descriptors"][:] = desc

def process_sequence(sequence, input_path, num_kp, suffix):
  """Process a single KITTI sequence"""
  detector = ExtractSIFT(num_kp)
  
  # Get image lists
  sequence_path = os.path.join(input_path, sequence, 'image_0')
  if not os.path.exists(sequence_path):
    print(f"Sequence path does not exist: {sequence_path}")
    return 0
    
  search = os.path.join(sequence_path, '*.png')
  listing = glob.glob(search)
  if not listing:
    print(f"No images found in {sequence_path}")
    return 0
    
  listing.sort()  # Sort to process images in sequential order

  # Create output directory if needed
  output_dir = os.path.join(input_path, sequence, 'features')
  os.makedirs(output_dir, exist_ok=True)

  print(f"Processing KITTI sequence {sequence} with {len(listing)} images...")
  
  for img_path in tqdm(listing):
    kp, desc = detector.run(img_path)
    # Extract just the filename without path
    img_name = os.path.basename(img_path)
    # Save to the features directory
    save_path = os.path.join(output_dir, img_name+'.'+suffix+'.hdf5')
    write_feature(kp, desc, save_path)
  
  print(f"Extracted features saved to {output_dir}")
  return len(listing)

if __name__ == "__main__":
  opt = parser.parse_args()
  
  # Process all specified sequences
  sequences = opt.sequences.split(',')
  total_images = 0
  
  print(f"Will process {len(sequences)} sequences: {sequences}")
  
  for sequence in sequences:
    num_processed = process_sequence(sequence, opt.input_path, opt.num_kp, opt.suffix)
    total_images += num_processed
    
  print(f"All done! Processed {total_images} images from {len(sequences)} sequences.")





