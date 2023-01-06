# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset.md_seq import MoDaSeq, paired_collate_fn

from utils.log import Logger
from utils.functional import str2bool, load_data, load_data_aist, check_data_distribution, visualizeAndWrite, \
    load_test_data_aist, load_test_data, dir_setting
from torch.optim import *
import warnings
from tqdm import tqdm
import itertools
import pdb
import numpy as np
import models
import datetime
import utils.build_util as build_util

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt


class GPT_CALL:
    def __init__(self, args):
        self.device = None
        self.config = args
        torch.backends.cudnn.benchmark = True
        # self._build()
        self.ckptdir, self.evaldir, self.gtdir, self.visdir, self.expdir = dir_setting(self.config)
        self.init_models, self.training_data, self.test_loader, self.dance_names, self.optimizer, self.schedular \
            = build_util.build(config=self.config)

    def train(self):
        vqvae = self.init_models[0].eval()
        gpt = self.init_models[1].train()

        config = self.config
        data = self.config.data
        training_data = self.training_data
        test_loader = self.test_loader
        optimizer = self.optimizer
        log = Logger(self.config, self.expdir)
        updates = 0

        checkpoint = torch.load(config.vqvae_weight)
        vqvae.load_state_dict(checkpoint['model'], strict=False)

        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight != '':
            print('Use pretrained model!')
            print(config.init_weight)
            checkpoint = torch.load(config.init_weight)
            gpt.load_state_dict(checkpoint['model'], strict=False)
        # self.model.eval()

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        # if args.cuda:
        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')

        # Training Loop
        for epoch_i in range(1, config.epoch + 1):
            log.set_progress(epoch_i, len(training_data))

            for batch_i, batch in enumerate(training_data):
                music_seq, pose_seq = batch

                music_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)
                pose_seq[:, :, :3] = 0
                optimizer.zero_grad()

                # music relative rate: ds_rate / music relative rate = music sample frequency / dance

                # 32, 40, 55

                with torch.no_grad():
                    quants_pred = vqvae.module.encode(pose_seq)
                    if isinstance(quants_pred, tuple):
                        quants_input = tuple(
                            quants_pred[ii][0][:, :-1].clone().detach() for ii in range(len(quants_pred)))
                        quants_target = tuple(
                            quants_pred[ii][0][:, 1:].clone().detach() for ii in range(len(quants_pred)))
                    else:
                        quants = quants_pred[0]
                        quants_input = quants[:, :-1].clone().detach()
                        quants_target = quants[:, 1:].clone().detach()
                output, loss = gpt(quants_input, music_seq, quants_target)

                loss.backward()

                # update parameters
                optimizer.step()

                stats = {
                    'updates': updates,
                    'loss': loss.item()
                }
                # if epoch_i % self.config.log_per_updates == 0:
                log.update(stats)
                updates += 1

            checkpoint = {
                'model': gpt.state_dict(),
                'config': config,
                'epoch': epoch_i
            }

            # Save checkpoint
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
            # Eval
            if epoch_i % config.test_freq == 0:
                with torch.no_grad():
                    print("Evaluation...")
                    gpt.eval()
                    results = []
                    random_id = 0  # np.random.randint(0, 1e4)
                    quants_out = {}
                    for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                        # Prepare data
                        music_seq, pose_seq = batch_eval

                        music_seq = music_seq.to(self.device)
                        pose_seq = pose_seq.to(self.device)

                        quants = vqvae.module.encode(pose_seq)
                        if isinstance(quants, tuple):
                            x = tuple(quants[i][0][:, :1] for i in range(len(quants)))
                        else:
                            x = quants[0][:, :1]

                        zs = gpt.module.sample(x, cond=music_seq,
                                               shift=config.sample_shift if hasattr(config, 'sample_shift') else None)
                        pose_sample = vqvae.module.decode(zs)

                        if config.global_vel:
                            global_vel = pose_sample[:, :, :3].clone()
                            pose_sample[:, 0, :3] = 0
                            for iii in range(1, pose_sample.size(1)):
                                pose_sample[:, iii, :3] = pose_sample[:, iii - 1, :3] + global_vel[:, iii - 1, :]

                        if isinstance(zs, tuple):
                            quants_out[self.dance_names[i_eval]] = tuple(
                                zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs)))
                        else:
                            quants_out[self.dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

                        results.append(pose_sample)

                    visualizeAndWrite(results, config, self.visdir, self.dance_names, epoch_i, quants_out)
                gpt.train()
            self.schedular.step()

    def eval(self):
        with torch.no_grad():
            vqvae = self.model.eval()
            gpt = self.model2.eval()

            config = self.config

            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)

            epoch_tested = config.testing.ckpt_epoch

            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation...")
            checkpoint = torch.load(ckpt_path)
            gpt.load_state_dict(checkpoint['model'])
            gpt.eval()

            results = []
            random_id = 0  # np.random.randint(0, 1e4)
            # quants = {}
            quants_out = {}
            for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                # Prepare data
                # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
                music_seq, pose_seq = batch_eval
                music_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)

                quants = vqvae.module.encode(pose_seq)

                if isinstance(quants, tuple):
                    x = tuple(quants[i][0][:, :1].clone() for i in range(len(quants)))
                else:
                    x = quants[0][:, :1].clone()

                if hasattr(config, 'random_init_test') and config.random_init_test:
                    if isinstance(quants, tuple):
                        for iij in range(len(x)):
                            x[iij][:, 0] = torch.randint(512, (1,))
                    else:
                        x[:, 0] = torch.randint(512, (1,))

                zs = gpt.module.sample(x, cond=music_seq,
                                       shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                pose_sample = vqvae.module.decode(zs)

                if config.global_vel:
                    global_vel = pose_sample[:, :, :3].clone()
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii - 1, :3] + global_vel[:, iii - 1, :]

                results.append(pose_sample)
                if isinstance(zs, tuple):
                    quants_out[self.dance_names[i_eval]] = tuple(
                        zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs)))
                else:
                    quants_out[self.dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

            visualizeAndWrite(results, config, self.evaldir, self.dance_names, epoch_tested, quants_out)

    def _build(self):
        config = self.config
        self.start_epoch = 0
        # self._dir_setting()
        self._build_model()
        if not (hasattr(config, 'need_not_train_data') and config.need_not_train_data):
            self._build_train_loader()
        if not (hasattr(config, 'need_not_test_data') and config.need_not_test_data):
            self._build_test_loader()
        self._build_optimizer()

    def _build_model(self):
        """ Define Model """
        config = self.config
        if hasattr(config.structure, 'name') and hasattr(config.structure_generate, 'name'):
            print(f'using {config.structure.name} and {config.structure_generate.name} ')
            model_class = getattr(models, config.structure.name)
            model = model_class(config.structure)

            model_class2 = getattr(models, config.structure_generate.name)
            model2 = model_class2(config.structure_generate)
        else:
            raise NotImplementedError("Wrong Model Selection")

        model = nn.DataParallel(model)
        model2 = nn.DataParallel(model2)
        self.model2 = model2.cuda()
        self.model = model.cuda()

    def _build_train_loader(self):

        data = self.config.data
        if data.name == "aist":
            print("train with AIST++ dataset!")
            external_wav_rate = self.config.ds_rate // self.config.external_wav_rate if hasattr(self.config,
                                                                                                'external_wav_rate') else 1
            external_wav_rate = self.config.music_relative_rate if hasattr(self.config,
                                                                           'music_relative_rate') else external_wav_rate
            train_music_data, train_dance_data, _ = load_data_aist(
                data_dir=data.train_dir,
                interval=data.seq_len,
                move=self.config.move if hasattr(self.config, 'move') else 8,
                rotmat=self.config.rotmat,
                external_wav=self.config.external_wav if hasattr(self.config, 'external_wav') else None,
                external_wav_rate=external_wav_rate,
                wav_padding=self.config.wav_padding * (
                        self.config.ds_rate // self.config.music_relative_rate) if hasattr(self.config,
                                                                                           'wav_padding') else 0)

        else:
            train_music_data, train_dance_data = load_data(
                data.train_dir,
                interval=data.seq_len,
                data_type=data.data_type)
        self.training_data = prepare_dataloader(train_music_data, train_dance_data, self.config.batch_size)

    def _build_test_loader(self):
        config = self.config
        data = self.config.data
        if data.name == "aist":
            print("test with AIST++ dataset!")
            music_data, dance_data, dance_names = load_test_data_aist(
                data.test_dir,
                move=config.move,
                rotmat=config.rotmat,
                external_wav=config.external_wav if hasattr(self.config, 'external_wav') else None,
                external_wav_rate=self.config.external_wav_rate if hasattr(self.config, 'external_wav_rate') else 1,
                wav_padding=self.config.wav_padding * (
                        self.config.ds_rate // self.config.music_relative_rate) if hasattr(self.config,
                                                                                           'wav_padding') else 0)

        else:
            music_data, dance_data, dance_names = load_test_data(
                data.test_dir)

        self.test_loader = torch.utils.data.DataLoader(
            MoDaSeq(music_data, dance_data),
            batch_size=1,
            shuffle=False
            # collate_fn=paired_collate_fn,
        )
        self.dance_names = dance_names

    def _build_optimizer(self):
        # model = nn.DataParallel(model).to(device)
        config = self.config.optimizer
        try:
            optim = getattr(torch.optim, config.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' + config.type)

        self.optimizer = optim(itertools.chain(self.model2.module.parameters(),
                                               ),
                               **config.kwargs)
        self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)

    def _dir_setting(self):
        data = self.config.data
        self.expname = self.config.expname
        self.experiment_dir = os.path.join("../GMWM/", "experiments")
        self.expdir = os.path.join(self.experiment_dir, self.expname)

        if not os.path.exists(self.expdir):
            os.mkdir(self.expdir)

        self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
        if not os.path.exists(self.visdir):
            os.mkdir(self.visdir)

        self.jsondir = os.path.join(self.visdir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir):
            os.mkdir(self.jsondir)

        self.histdir = os.path.join(self.visdir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir):
            os.mkdir(self.histdir)

        self.imgsdir = os.path.join(self.visdir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir):
            os.mkdir(self.imgsdir)

        self.videodir = os.path.join(self.visdir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir):
            os.mkdir(self.videodir)

        self.ckptdir = os.path.join(self.expdir, "ckpt")
        if not os.path.exists(self.ckptdir):
            os.mkdir(self.ckptdir)

        self.evaldir = os.path.join(self.expdir, "eval")
        if not os.path.exists(self.evaldir):
            os.mkdir(self.evaldir)

        self.gtdir = os.path.join(self.expdir, "gt")
        if not os.path.exists(self.gtdir):
            os.mkdir(self.gtdir)

        self.jsondir1 = os.path.join(self.evaldir, "jsons")  # -- imgs, videos, jsons
        if not os.path.exists(self.jsondir1):
            os.mkdir(self.jsondir1)

        self.histdir1 = os.path.join(self.evaldir, "hist")  # -- imgs, videos, jsons
        if not os.path.exists(self.histdir1):
            os.mkdir(self.histdir1)

        self.imgsdir1 = os.path.join(self.evaldir, "imgs")  # -- imgs, videos, jsons
        if not os.path.exists(self.imgsdir1):
            os.mkdir(self.imgsdir1)

        self.videodir1 = os.path.join(self.evaldir, "videos")  # -- imgs, videos, jsons
        if not os.path.exists(self.videodir1):
            os.mkdir(self.videodir1)

        self.sampledir = os.path.join(self.evaldir, "samples")  # -- imgs, videos, jsons
        if not os.path.exists(self.sampledir):
            os.mkdir(self.sampledir)


def prepare_dataloader(music_data, dance_data, batch_size):
    data_loader = torch.utils.data.DataLoader(
        MoDaSeq(music_data, dance_data),
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return data_loader
