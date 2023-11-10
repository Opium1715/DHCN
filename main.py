import argparse
import datetime
import pickle
from utils.dataloader import compute_item_num, DataLoader, compute_max_node
from tensorflow import keras
from model import DHCN
from utils.myCallback import HistoryRecord, P_MRR
from utils.loss import Loss_with_L2

parser = argparse.ArgumentParser()
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.02, help='ssl task maginitude')
opt = parser.parse_args()
print(opt)

train_data = pickle.load(open("dataset/tmall/train.txt", 'rb'))
test_data = pickle.load(open("dataset/tmall/test.txt", "rb"))
all_train_data = pickle.load(open("dataset/tmall/all_train_seq.txt", 'rb'))
item_num = compute_item_num(all_train_data)  # 40727
epoch_steps = len(train_data[1]) / 100
test_data_size = len(test_data[1])
train_dataloader = DataLoader(train_data, n_node=item_num, train_mode=True).dataloader()
test_dataloader = DataLoader(test_data, n_node=item_num, train_mode=False).dataloader()
adj = train_data.get_adj()

# MODEL
save_dir = 'logs'
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
model = DHCN(adj, item_num, layers=opt.layer, beta=opt.beta)
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opt.lr,
                                                          decay_rate=opt.lr_dc,
                                                          decay_steps=opt.lr_dc_step * epoch_steps,
                                                          staircase=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='MRR@20',
                                               min_delta=0,
                                               patience=5,
                                               verbose=1,
                                               mode='max')
history_recoder = HistoryRecord(log_dir=os.path.join(save_dir, 'log_' + time_str))
p_mrr = P_MRR(val_data=test_dataloader, performance_mode=2, val_size=int(test_data_size/100))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=Loss_with_L2(model=model, l2=opt.l2),
              run_eagerly=False)
model.fit(x=train_dataloader,
          epochs=30,
          verbose=1,
          callbacks=[p_mrr, history_recoder, early_stopping],
          validation_data=test_dataloader)

