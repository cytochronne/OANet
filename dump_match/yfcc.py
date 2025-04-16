import argparse
from dataset import Dataset

def str2bool(v):
    return v.lower() in ("true", "1")

# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--raw_data_path', type=str, default='../raw_data/yfcc100m/raw_data/',
  help='raw data path. default:../raw_data/')
parser.add_argument('--dump_dir', type=str, default='../data_dump/',
  help='data dump path. default:../data_dump')
parser.add_argument('--desc_name', type=str, default='sift-1000',
  help='prefix of desc filename, default:sift-1000')
parser.add_argument('--vis_th', type=int, default=50,
  help='visibility threshold')
parser.add_argument('--pair_num', type=int, default=1000,
  help='pair num. 1000 for test seq')

if __name__ == "__main__":
    config = parser.parse_args()

    # ✅ 只处理 big_ben_1 作为 test 数据
    test_seqs = ['big_ben_1']
    print('test seq len ' + str(len(test_seqs)))

    yfcc_te = Dataset(
        config.raw_data_path + 'yfcc100m/',
        config.dump_dir,
        'yfcc-' + config.desc_name + '-test.hdf5',
        test_seqs,
        'test',
        config.desc_name,
        config.vis_th,
        config.pair_num,
        None  # 这个可以是 None 也可以写路径
    )

    # ❌ 注释掉不需要的 train/val 部分
    # with open('yfcc_train.txt','r') as ofp:
    #     train_seqs = ofp.read().split('\n')
    # if len(train_seqs[-1]) == 0:
    #     del train_seqs[-1]
    # print('train seq len ' + str(len(train_seqs)))
    # yfcc_tr_va = Dataset(...)
    # yfcc_tr_tr = Dataset(...)
