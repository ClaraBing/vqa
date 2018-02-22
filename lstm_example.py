import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import h5py
from time import time
import sklearn.metrics as skm
from random import randint


SHAPE_DEBUG = False
PERFORM_TRAIN = False
# Hyperparam
hidden_size = 200
train_seq_len = 64
test_seq_len = 128
use_combined = True
temp_aug = False
bidirectional = True
# Training param
num_act = 4
lr = 1e-4
orig_lr = 1e-4
weight_decay = 5e-4
sigmoid = nn.Sigmoid()
print_step = 20
gpu_dtype = torch.cuda.FloatTensor

class H5pyFeatureLoader(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.targets = labels

    def __getitem__(self, index):
        """
        Args:    index (int): Index
        Returns: tuple: (image, target) where target is class_index of the target class.
        """
        target_vec = np.zeros((self.targets.shape[1], num_act))
        for i in range(test_seq_len):
          if self.targets[index, i] != 0:
            target_vec[i, self.targets[index, i]-1] = 1
        return {'features':self.features[index], 'targets':target_vec}

    def __len__(self):
        return self.features.shape[0]

# Input / Output dir
input_base_dir = '/home/bingbin/debug/'
h5py_features = input_base_dir + 'b256_lr0.000100_wd0.000500_momen0.9_cam_random/feats_cam_random_v1_cls4_with_flip.h5'
h5py_labels = input_base_dir + 'labels_cam_random_v1_cls4_with_flip.h5'
out_dir = 'lstm_cam_random_cls4_hidden{:d}_lr{:f}_with_flip{:s}{:s}{:s}/'.format(
    hidden_size, orig_lr, '_bidirect' if bidirectional else '', '_combined' if use_combined else '', '_temp{:d}'.format(train_seq_len) if temp_aug else '')
if not os.path.exists(out_dir):
  os.mkdir(out_dir)
else:
  print('WARNING: {:s} already exists. Models may be overwritten.'.format(out_dir))


# Load features
h5feats = h5py.File(h5py_features, 'r')
h5labels = h5py.File(h5py_labels, 'r')
if use_combined:
  # Training using train + val
  train_batch_size = 16
  dataset_train = H5pyFeatureLoader(np.concatenate([h5feats['train'], h5feats['val']]), np.concatenate([h5labels['train'], h5labels['val']]))
  train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, drop_last=True)
  print('train_loader len (train & val combined):', len(train_loader))
  val_loader = None
else:
  # Train
  train_batch_size = 15
  dataset_train = H5pyFeatureLoader(h5feats['train'], h5labels['train'])
  train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)
  print('train_loader len:', len(train_loader))
  # Val
  val_batch_size = 1
  dataset_val = H5pyFeatureLoader(h5feats['val'], h5labels['val'])
  val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=val_batch_size, shuffle=False)
  print('val_loader len:', len(val_loader))

# Test
val_batch_size = 1
train_test_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
dataset_test = H5pyFeatureLoader(h5feats['test'], h5labels['test'])
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)
print('test_loader len:', len(test_loader))


def train(lstm, linear, loss_fn, optimizer_lstm, optimizer_linear, hidden_size, epoch):
    lstm.train()
    linear.train()
    
    num_correct = 0
    num_samples = 0
    per_class_correct = np.zeros(num_act)
    per_class_samples = np.zeros(num_act)
    
    total_loss = 0

    epoch_time_start = time()
    for batch_id, item in enumerate(train_loader):
        features = Variable(item['features'].transpose(1,0)).type(gpu_dtype).cuda()
        targets = Variable(item['targets']).type(gpu_dtype).cuda()
        # the following 2 lines serve the same purpose as init_hidden() in your example
        h0 = Variable(torch.rand(2*(2 if bidirectional else 1), train_batch_size, hidden_size)).type(gpu_dtype)
        c0 = Variable(torch.rand(2*(2 if bidirectional else 1), train_batch_size, hidden_size)).type(gpu_dtype)

        iter_time_start = time()
        scores, ht = lstm(features, (h0, c0))
        scores = linear(scores) # keep outputs from every frame in the sequence
       
        if SHAPE_DEBUG:
            print('features shape:', features.shape)
            print('scores shape:', scores.shape)
            print('target shape:', targets.shape)
        
        scores_reshaped = scores.transpose(1,0).contiguous().view(-1, scores.size(2))
        targets_reshaped = targets.view(-1, targets.size(2))
        if SHAPE_DEBUG:
            print('scores_reshaped:', scores_reshaped.size())
            print('targets_reshaped:', targets_reshaped.size())
        loss = loss_fn(scores_reshaped, targets_reshaped)
        with open("loss.txt", "a") as myfile:
            myfile.write('(%.2f),' % (loss.data[0]))

        optimizer_lstm.zero_grad()
        optimizer_linear.zero_grad()
        # NOTE: backward is different from what I said before;
        # I said we need to also do model.backward(), but we don't (i.e. we only need to call backward once)
        # since autograd (i.e. Variable) already handles the backprop for us.
        loss.backward()
        optimizer_lstm.step()
        optimizer_linear.step()
       
        # loss
        loss_cpu = loss.data[0]
        total_loss += loss_cpu

        # accuracy
        preds = (scores.transpose(1,0) > 0.5).data # NOTE: currently using 0.5 for all classes, which may not be a good threshold
        targets_bytes = targets.data.type(torch.cuda.ByteTensor)
        num_correct += (preds == targets_bytes).sum()
        num_samples += preds.numel()
        for cid in range(num_act):
            per_class_correct[cid] += (preds[:,:,cid] == targets_bytes[:,:,cid]).sum()
            per_class_samples[cid] += targets_bytes.size(0) * targets_bytes.size(1)

        # print('time per batch: {:f}'.format(time()-iter_time_start))
        
    # Below are some irrelevant logging stuff
    # NOTE: accuracy is not a good measure, since the dataset is dominated by negative classes
    acc = float(num_correct) / num_samples
    avg_loss = total_loss/len(train_loader)
    print('Train({:d}): loss:{:.5f}; accu: {:.2f} ({:d}/{:d}) (avg time: {:.3f}s)'.format(
        epoch, avg_loss, 100*acc, num_correct, num_samples, (time()-epoch_time_start)/len(train_loader)))

    with open("{:s}/output_lstm_{:f}.txt".format(out_dir, orig_lr), "a") as myfile:
        myfile.write('Train({:d}): loss:{:.5f}; accu:{:.2f} ({:d}/{:d})\n'.format(epoch, avg_loss, 100*acc, num_correct, num_samples))
    with open("{:s}/train_lstm_{:f}.txt".format(out_dir, orig_lr), "a") as myfile:
        myfile.write('(%.2f),\n' % (100 * acc))


def val(lstm, linear, hidden_size, data_loader, epoch=0, save=False, split='unspecified'):
    lstm.eval()
    linear.eval()

    num_correct = 0
    num_samples = 0

    total_loss = 0 
    per_class_correct = np.zeros(num_act)
    per_class_samples = np.zeros(num_act)
    GT = []
    PREDS = []
    for batch_id, item in enumerate(data_loader):
        features = Variable(item['features'].transpose(1,0)).type(gpu_dtype).cuda()
        targets = Variable(item['targets']).type(gpu_dtype).cuda()
        h0 = Variable(torch.rand(2*(2 if bidirectional else 1), val_batch_size, hidden_size)).type(gpu_dtype)
        c0 = Variable(torch.rand(2*(2 if bidirectional else 1), val_batch_size, hidden_size)).type(gpu_dtype)

        scores, ht = lstm(features, (h0, c0))
        scores = linear(scores)
        scores = scores.transpose(1,0)

        scores_reshaped = scores.contiguous().view(-1, scores.size(2))
        targets_reshaped = targets.view(-1, targets.size(2))
        loss = loss_fn(scores_reshaped, targets_reshaped)
        total_loss += loss.data[0]

        probs = sigmoid(scores)
        preds = (probs > 0.5).data
        targets_bytes = targets.data.type(torch.cuda.ByteTensor)

        num_correct += (preds == targets_bytes).sum()
        num_samples += preds.numel()
        for cid in range(num_act):
            per_class_correct[cid] += (preds[:,:,cid] == targets_bytes[:,:,cid]).sum()
            per_class_samples[cid] += targets_bytes.size(0) * targets_bytes.size(1)
        GT += targets_bytes.type(torch.FloatTensor).numpy(),
        PREDS += probs.data.type(torch.FloatTensor).numpy(),

    acc = float(num_correct) / num_samples
    avg_loss = total_loss/len(data_loader)
    print('loss: %.6f; accu: %.2f (%d/%d)' % (avg_loss, 100*acc, num_correct, num_samples))

    # AP
    total_ap = 0
    ap_strs = []
    GT = np.concatenate(GT).reshape(-1, num_act)
    PREDS = np.concatenate(PREDS).reshape(-1, num_act)
    # NOTE: these two lines are temporary
    np.save(out_dir+'GT_test.npy', GT)
    np.save(out_dir+'PREDS_test.npy', PREDS)

    for act in range(num_act):
        precision, recall, thresh = skm.precision_recall_curve(GT[:, act].reshape(-1), PREDS[:, act].reshape(-1))
        for pi in range(1, len(precision)):
            if precision[pi] < precision[pi-1]:
                precision[pi] = precision[pi-1]
      
        ap = skm.auc(recall, precision)
        total_ap += ap
        ap_strs += '{:d}:{:4f}'.format(act, ap),

    # print out per class AP & mean AP
    print('AP:', ' / '.join(ap_strs), '/', total_ap/num_act)

    # NOTE: temporary; uncomment this!!
    with open("{:s}/output_lstm_{:f}.txt".format(out_dir, orig_lr), "a") as myfile:
        myfile.write('Val ({:d}): {:d} / {:d} correct {:.2f})\n'.format(epoch, num_correct, num_samples, 100 * acc))
        myfile.write("--\n")
    with open("{:s}/val_lstm_{:f}.txt".format(out_dir, orig_lr), "a") as myfile:
        myfile.write('{:s}({:d}), ({:.2f}),{:s} / mean AP: {:f}\n'.format(split, epoch, 100 * acc, ' / '.join(ap_strs), total_ap / num_act))
    
    return avg_loss


if __name__ == '__main__':
    # pretrained_path = ''
    pretrained_path = 'lstm_cam_random_cls4_hidden200_lr0.000100_with_flip__bidirct_combined/lstm_pil_weighted_hidden_200_lr_0.0001_epoch_480'
    if pretrained_path:
        model_dict = torch.load(pretrained_path, map_location=lambda storage,loc:storage)
        lstm, linear = model_dict['lstm'], model_dict['linear']
        lstm = lstm.cuda()
        linear = linear.cuda()
        if 'optimizer_lstm' in model_dict:
            optimizer_lstm = model_dict['optimizer_lstm']
            optimizer_linear = model_dict['optimizer_linear']
        else:
            optimizer_lstm = optim.Adam(lstm.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=weight_decay)
            optimizer_linear = optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        lstm = nn.LSTM(512, hidden_size, 2, bidirectional=bidirectional).type(gpu_dtype) #input size, hidden size, num layers
        linear = nn.Linear(hidden_size*(2 if bidirectional else 1), num_act).type(gpu_dtype)
        optimizer_lstm = optim.Adam(lstm.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=weight_decay)
        optimizer_linear = optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)
   
    # loss function: you can choose your own; see torcy website for a list of loss funcs
    loss_fn = nn.MultiLabelSoftMarginLoss().type(gpu_dtype)

    # ==== Training Start ==== #
    if PERFORM_TRAIN:
        best_val = 100
        train_start = time()
        for i in range(0, 100):
            # each iteration in this for loop is an epoch
            print('Epoch ', i+1)
            train(lstm, linear, loss_fn, optimizer_lstm, optimizer_linear, hidden_size, i)

            if i % print_step == 0:
                # Check model performance on training set
                print('Train accu:')
                val(lstm, linear, hidden_size, train_test_loader, epoch=i, save=True, split='train')

                if val_loader:
                    # Chech model performance on validation set
                    print('Val accu:')
                    curr_loss = val(lstm, linear, hidden_size, val_loader, epoch=i, save=True, split='val')
                    if save and curr_loss < best_val:
                        best_val = curr_loss
                        torch.save({'lstm':lstm, 'linear':linear, 'optimizer':optimizer}, "{:s}/best_lstm_pil_weighted_hidden_{:d}_lr_{:s}".format(out_dir, hidden_size, str(orig_lr)))
                        print('\nTraining time:', time()-train_start)
                if i != 0:
                    torch.save({'lstm':lstm, 'linear':linear, 'optimizer_lstm':optimizer_lstm, 'optimizer_linear':optimizer_linear}, "{:s}/lstm_pil_weighted_hidden_{:d}_lr_{:s}_epoch_{:d}".format(out_dir, hidden_size, str(orig_lr), i))
    # ==== Training End ==== #


    # ==== Testing ==== #
    print('\n\nTest:')
    val(lstm, linear, hidden_size, test_loader, save=False, split='test')