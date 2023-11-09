import argparse
import pickle
from utils.dataloader import compute_item_num, DataLoader, compute_max_node

parser = argparse.ArgumentParser()
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.02, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
opt = parser.parse_args()
print(opt)

train_data = pickle.load(open("dataset/tmall/train.txt", 'rb'))
test_data = pickle.load(open("dataset/tmall/test.txt", "rb"))
all_train_data = pickle.load(open("dataset/tmall/all_train_seq.txt", 'rb'))
item_num = compute_item_num(all_train_data)  # 40727

train_data = DataLoader(train_data, n_node=item_num, train_mode=True)
# test_data = DataLoader(test_data, train_mode=False).dataloader()

# MODEL
