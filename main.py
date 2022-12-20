import os
import sys
import pickle
import argparse
import time

from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch

from utils import util, valid_angle_check, functional
from dataset.md_seq import MoDaSeq, paired_collate_fn

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import librosa
import librosa.display


def prepare_dataloader(music_data, dance_data, batch_size):
    data_loader = torch.utils.data.DataLoader(
        MoDaSeq(music_data, dance_data),
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    return data_loader


def _build_train_loader(path):
    train_dir = path
    seq_len = 240
    print("train with AIST++ dataset!")
    external_wav_rate = 1
    train_music_data, train_dance_data = functional.load_data_aist(
        data_dir=train_dir,
        interval=seq_len,
        move=8,
        rotmat=False,
        external_wav=None,
        external_wav_rate=external_wav_rate,
        wav_padding=0)

    training_data = prepare_dataloader(music_data=train_music_data, dance_data=train_dance_data, batch_size=1)
    return training_data


def plt_3d_dct(y):
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(y, alpha=0.5, x_axis='hz', y_axis='log')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    dct_m, idct_m = util.get_dct_matrix(240)
    dct_m_all = dct_m.float()
    idct_m_all = idct_m.float()
    print(dct_m_all.shape)

    path = "data/aistpp_train_wav"
    training_data = _build_train_loader(path)
    for batch_i, batch in enumerate(training_data):
        music_seq, pose_seq = batch

        pose_seq = pose_seq.reshape(-1, 240, 24, 3)

        transform = pose_seq.transpose(1, 2).transpose(2, 3)
        dct = torch.matmul(transform, dct_m).transpose(2, 3).transpose(1, 2)
        print(dct.shape)
        root_dct_x = dct[0, :, 0, 2]
        print(root_dct_x.numpy().shape)

        plt.figure(figsize=(8, 4))
        plt.subplot(211)
        plt.plot(pose_seq[0, :, 0, 2])
        plt.xlabel('sample')  # x轴样点
        plt.title('signal', fontsize=12, color='black')  # 标题名称、字体大小、颜色
        plt.subplot(212)
        plt.plot(root_dct_x)
        plt.xlabel('sample')  # x轴样点
        plt.title('dct', fontsize=12, color='black')  # 标题名称、字体大小、颜色
        plt.subplots_adjust(hspace=0.6)
        plt.show()
        exit()

