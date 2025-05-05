import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import cv2
import h5py
import collections

def str2bool(v):
    return v.lower() in ("true", "1")
    
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, default='../../ETHdataset',
  help='Dataset directory (default: ../../Datasets_filtered).')
parser.add_argument('--img_dir', type=str, default='images/dslr_images_undistorted',
  help='Directory within input_path containing images (default: images/dslr_images_undistorted).')
parser.add_argument('--img_ext', type=str, default='.jpg',
  help='Image file extension (default: .jpg).')
parser.add_argument('--num_kp', type=int, default='1000',
  help='keypoint number, default:1000')
parser.add_argument('--suffix', type=str, default='sift-1000',
  help='suffix of filename, default:sift-1000')

class ExtractSIFT(object):
  def __init__(self, num_kp, contrastThreshold=1e-5):
    self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

  def run(self, img_path):
    img = cv2.imread(img_path)
    if img is None:
      print(f"Failed to load image: {img_path}")
      return None, None
      
    # SIFT feature extraction
    cv_kp, desc = self.sift.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc

def write_feature(pts, desc, filename):
  with h5py.File(filename, "w") as ifp:
      ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
      ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
      ifp["keypoints"][:] = pts
      ifp["descriptors"][:] = desc

def process_images(input_path, img_dir, img_ext, num_kp, suffix):
  """Process images from a directory in COLMAP format"""
  detector = ExtractSIFT(num_kp)
  
  # Create full image directory path
  image_path = os.path.join(input_path, img_dir)
  if not os.path.exists(image_path):
    print(f"Image directory does not exist: {image_path}")
    return 0
    
  # Find all jpg images (case insensitive)
  search_pattern = '*' + img_ext.lower()
  jpg_files = glob.glob(os.path.join(image_path, search_pattern))
  jpg_files.extend(glob.glob(os.path.join(image_path, search_pattern.upper())))
  
  if not jpg_files:
    print(f"No {img_ext} images found in {image_path}")
    return 0
    
  jpg_files.sort()  # Sort to process images in sequential order

  # Create output directory for features
  output_dir = os.path.join(input_path, 'features')
  os.makedirs(output_dir, exist_ok=True)

  print(f"Processing {len(jpg_files)} images from {image_path}...")
  
  for img_path in tqdm(jpg_files):
    kp, desc = detector.run(img_path)
    if kp is None or desc is None:
      continue
      
    # Extract just the filename without path and extension
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Save to the features directory
    save_path = os.path.join(output_dir, f"{img_name}.{suffix}.hdf5")
    write_feature(kp, desc, save_path)
  
  print(f"Extracted features saved to {output_dir}")
  return len(jpg_files)

if __name__ == "__main__":
  opt = parser.parse_args()
  
  # Process all images
  num_processed = process_images(
      opt.input_path, 
      opt.img_dir, 
      opt.img_ext,
      opt.num_kp, 
      opt.suffix
  )
    
  print(f"All done! Processed {num_processed} images.")