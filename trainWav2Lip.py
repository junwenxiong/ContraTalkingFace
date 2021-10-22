import os
from numba import config
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as data_utils
from dataset import Dataset as DD
from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual, VGGLoss, Wav2Lip_ori, Sync_loss
from utils.checkpoint import load_checkpoint, save_checkpoint, save_sample_images
from options.train_options import TrainOptions
from options.test_options import TestOptions
from wav2lip_net import W2LNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

syncnet_T = 5
syncnet_mel_step_size = 16

if __name__ == "__main__":
    args, configLogger = TrainOptions().parse()

    # Dataset and Dataloader setup
    train_dataset = DD(args, 'train', )
    test_dataset = DD(args, 'val', )

    train_data_loader = data_utils.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers)

    test_data_loader = data_utils.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=4)

    # Model
    model = W2LNet(args, configLogger)

    for epoch in range(args.nepochs):
        model.train_network(train_data_loader, test_data_loader)
