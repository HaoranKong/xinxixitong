import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class csiDatasets(Dataset):
    def __init__(self, config, mode="train"):
        self.mode = mode
        self.config = config
        self.train_set = torch.tensor(np.load(os.path.join(self.config.data_dir, "train.npy")), dtype=torch.float32)
        self.val_set = torch.tensor(np.load(os.path.join(self.config.data_dir, "val.npy")), dtype=torch.float32)
        self.test_set = torch.tensor(np.load(os.path.join(self.config.data_dir, "test.npy")), dtype=torch.float32)
        if mode == "train":
            self.csi_set = self.train_set        # training set
            self.length = self.train_set.shape[0]
        elif mode == "val":
            self.csi_set = self.val_set  # training set
            self.length = self.val_set.shape[0]
        else:
            self.csi_set = self.test_set  # training set
            self.length = self.test_set.shape[0]

        self.csi_channel, self.csi_height, self.csi_width = config.csi_dims

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        csi = self.csi_set[idx]
        csi_reshape = csi.view(self.csi_channel, self.csi_height, self.csi_width)      # 2 * 32 * 32

        # The block for normalizing the angular-delay csi matrix (need to be filled)
        # 角度时延域csi矩阵的归一化板块（需要被填充）
        #
        #
        #
        # csi_norm = ??????                                                             # 归一化矩阵的维度同样是2 * 32 * 32

        return csi_norm


def get_loader(config):
    train_dataset = csiDatasets(config, mode="train")
    val_dataset = csiDatasets(config, mode="val")
    # test_dataset = csiDatasets(config, mode="test")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=1,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)

    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=config.batch_size,
    #                                           shuffle=False)

    return train_loader, val_loader
