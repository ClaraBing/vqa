import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.models as models

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # NOTE: this is required; otherwise loading truncated images will give errors
# example usage: Image.open(open(img_path, 'rb')).convert('RGB')

import numpy as np
import os
import sys
import time
from glob import glob
import cv2
import h5py
import pickle
import json


def extract_features(split):
  img_dir_base = './'

  transform = transforms.Compose([
                transforms.Resize([224, 224]),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              ])

  # load model
  model = models.resnet101(pretrained=True) # NOTE: VGG is used in DiDeMo (not sure which version though)
  model = model.cuda()
  model.eval()

  def get_feature_hook(self, input, output):
    if output.data.size(0) == 1:
      out = output.data.cpu().numpy().reshape(output.data.size(1))
    else:
      out = output.data.cpu().numpy().reshape(output.data.size(0), output.data.size(1))
    feats.append(out)

  feat_layer = model._modules.get('avgpool')
  feat_layer.register_forward_hook(get_feature_hook)
  feat_file = '/home/grapefruit/vqa/resnet101_avgpool.h5'.format(split)
  featsh5 = h5py.File(feat_file, 'w')

  print_every = 500
  cache_every = 2000
  start = time.time()
  img_paths = sorted(glob('v7w_*.jpg'))
  all_feats = []
  for i, img in enumerate(img_paths):
    if i and i%print_every==0:
      avg_time = (time.time() - start)/print_every
      print('Processing {:d}/{:d} (avg: {:f}s)'.format(i, len(img_paths), avg_time))
      sys.stdout.flush()
      start = time.time()

    feats = []
    img = Image.open(open(path, 'rb'))
    img = transform(img)
    img_var = Variable(torch.unsqueeze(img, 0)).cuda()
    out = model.forward(img_var.un)
    all_feats += feats[0],

    if i and i%cache_every==0:
      with open('/home/grapefruit/vqa/all_feats_tmp.pickle', 'wb') as handle:
        pickle.dump(all_feats, handle)
    # feats = np.concatenate(feats, axis=0)
    # print('type feats:', type(feats))
    # print('feats size:', feats.shape)
    # feats_arr = np.asarray(feats).squeeze()
    # print('feats_arr type:', type(feats_arr))
    featsh5.create_dataset(img[img.find('_')+1:img.find('.')], data=feats[0])
  featsh5.close()


if __name__ == '__main__':
  extract_features()
