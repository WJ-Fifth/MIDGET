import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from dataset.md_seq import MoDaSeq

from dataset.md_seq_ac import MoDaSeqAC

from utils.log import Logger
from utils.functional import visualizeAndWrite, load_test_data_aist, load_data_aist, dir_setting
import warnings
from tqdm import tqdm
import itertools
import numpy as np
import models
import datetime

import json
import matplotlib.pyplot as plt
import utils.build_util as build_util

warnings.filterwarnings('ignore')

music_root = None


def get_beat(key, music_root):
    path = os.path.join(music_root, key)
    with open(path) as f:
        sample_dict = json.loads(f.read())
        beats = np.array(sample_dict['music_array'])[:, 53]

        return beats


class Actor_Critic:
    def __init__(self, args):
        self.device = None
        self.config = args
        torch.backends.cudnn.benchmark = True
        # self._build()
        self.ckptdir, self.evaldir, self.gtdir, self.visdir, self.expdir = dir_setting(self.config)
        self.init_models, self.training_data, self.test_loader, self.dance_names, self.optimizer, self.schedular \
            = build_util.build(config=self.config)

        model_reward = getattr(models, args.reward.name)
        reward = model_reward(args.reward)
        dance_reward = nn.DataParallel(reward)
        self.dance_reward = dance_reward.cuda()

    def train(self):
        vqvae = self.init_models[0].eval()
        gpt = self.init_models[1].train()
        gpt.module.freeze_drop()

        config = self.config
        # training_data = self.training_data
        training_data = self.test_loader
        test_loader = self.test_loader
        optimizer = self.optimizer
        log = Logger(self.config, self.expdir)
        updates = 0

        checkpoint = torch.load(config.vqvae_weight)
        vqvae.load_state_dict(checkpoint['model'], strict=False)

        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight != '':
            print('Use pretrained model!')
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

            # At the very beginning, generate the motion as test
            dance_up_seqs = []
            dance_down_seqs = []
            music_seqs = []
            beat_seqs = []
            for batch_i, batch in enumerate(training_data):
                music_seq, pose_seq = batch
                music_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)

                pose_seq[:, :, :3] = 0
                x = None

                # 1. generate motion on whole music (no grad)
                # NOTE the generation here should be consistent with the evaluation process (generate whole piece)
                with torch.no_grad():
                    quants_pred = vqvae.module.encode(pose_seq)
                    if isinstance(quants_pred, tuple):
                        quants = tuple(
                            quants_pred[ii][0][:, :-1].clone().detach() for ii in range(len(quants_pred)))
                        x = tuple(quants_pred[i][0][:, :1] for i in range(len(quants_pred)))
                    else:
                        quants = quants_pred[0]
                        x = quants_pred[0][:, :1]

                    gpt.eval()
                    zs = gpt.module.sample(x, cond=music_seq,
                                           shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                    dance_up_seqs.append(zs[0][0][0].data.cpu().numpy())
                    dance_down_seqs.append(zs[1][0][0].data.cpu().numpy())
                    music_seqs.append(music_seq[0].data.cpu().numpy())
                    beat_seqs.append(get_beat(self.dance_names[batch_i], config.rl_music_root))

            # 2. sample music-motion pair from generated data
            train_data = prepare_dataloader(music_seqs, (dance_up_seqs, dance_down_seqs), beat_seqs,
                                            self.config.batch_size, self.config.structure_generate.block_size + 1)

            log.set_progress(epoch_i, len(train_data))

            # 3. for each batch
            for batch_i, batch in enumerate(train_data):
                music_seq, pose_seq_up, pose_seq_down, beat_seq, mask_seq = batch
                music_seq = music_seq.to(self.device)[:, 1:]  # music (1..29)
                pose_seq_up = pose_seq_up.to(self.device)
                pose_seq_down = pose_seq_down.to(self.device)
                beat_seq = beat_seq.to(self.device)
                mask_seq = mask_seq.to(self.device)

                quants_pred = (pose_seq_up, pose_seq_down)

                # pose_seq[:, :, :3] = 0
                if isinstance(quants_pred, tuple):
                    # quants_input 0..28 len 29
                    quants_input = tuple(quants_pred[ii][:, :-1].clone().detach() for ii in range(len(quants_pred)))
                    # quants_output 1..29 len 29
                    quants_target = tuple(quants_pred[ii][:, 1:].clone().detach() for ii in range(len(quants_pred)))
                    # rewards_input 1..28 len 28
                    rewards_input = tuple([quants_pred[ii][:, 1:-1].clone().detach()] for ii in range(len(quants_pred)))
                    # actor input 0..27 len 28
                    quants_actor_input = tuple(
                        quants_pred[ii][:, :-2].clone().detach() for ii in range(len(quants_pred)))
                    # actor output 1..28 len 28
                    quants_actor_output = tuple(
                        quants_pred[ii][:, 1:-1].clone().detach() for ii in range(len(quants_pred)))

                else:
                    quants = quants_pred[0]
                    quants_input = quants[:, :-1].clone().detach()
                    quants_target = quants[:, 1:].clone().detach()
                    rewards_input = quants[:, 1:-1].clone().detach()
                    quants_actor_input = quants[:, :-2].clone().detach()
                    quants_actor_output = quants[:, 1:-1].clone().detach()

                pose_sample = vqvae.module.decode(rewards_input)
                # pose_sample [1...28] len 28

                # 3a. compute rewards from motion (1..28, len 28) and music (1*8..28*8, len 28*8)
                # print("Music beats: ", beat_seq[:, 8:-8].shape)
                # print("Motion sample: ", pose_sample.shape)
                rewards = self.dance_reward(pose_sample, beat_seq[:, 8:-8], config.ds_rate)
                # reward of action 0 ... 27 (dance 1...28, with music 1...28), len 28
                # print("Reward: ", rewards.shape)

                gpt.train()
                gpt.module.freeze_drop()
                optimizer.zero_grad()

                # 3b. If training actor net, then compute TDerror, without grad and cross_entropy_loss
                # 3c. if training critic net, then only compute TDerror, with grad

                values = gpt.module.critic(quants_input, music_seq)[:, :, 0]  # value of state [0 ... 28]

                # values[1..28] - values[0..27], len 28
                td_error = (rewards + config.gamma * values[:, 1:]).clone().detach() - values[:, :-1]

                with torch.no_grad():
                    gpt.eval()
                    output, actor_loss, entropy = gpt.module.actor(quants_actor_input, music_seq[:, :-1],
                                                                   quants_actor_output,
                                                                   reduction=False)  # output dance 1...28
                    gpt.train()
                    gpt.module.freeze_drop()

                # if you need entropy loss;
                # entropy loss is a common regularization in RL, but we don't use finally
                if hasattr(config, 'entropy_alpha'):
                    alpha = config.entropy_alpha
                    td_error = td_error.view(-1) + alpha * entropy.clone().detach()
                else:
                    alpha = 0
                    entropy = torch.zeros(td_error.view(-1).size()).cuda()

                # if training actor net:
                if epoch_i >= config.pretrain_critic_epoch and (
                        batch_i % (config.critic_iter + config.actor_iter) < config.actor_iter):
                    # print("training actor net")
                    output, actor_loss, entropy = gpt.module.actor(quants_actor_input, music_seq[:, :-1],
                                                                   quants_actor_output,
                                                                   reduction=False)  # output dance 1...28

                    loss = torch.sum(
                        (actor_loss * td_error.view(-1).clone().detach() - alpha * entropy) * mask_seq.view(
                            -1).clone().detach()) / torch.sum(mask_seq).clone().detach() * config.actor_loss_decay

                    actor_loss = torch.sum(actor_loss * mask_seq.view(-1).clone().detach()) / torch.sum(
                        mask_seq).clone().detach()
                # if training critic net:
                else:
                    # print("training critic net")
                    loss = torch.mean(td_error ** 2)

                actor_loss = actor_loss.clone().detach().mean()
                loss.backward()

                # update parameters
                optimizer.step()
                # exit()
                stats = {
                    'updates': updates,
                    'reward': ((rewards.view(-1) * mask_seq.view(-1)).sum() / mask_seq.sum()).detach().clone().item(),
                    'TD-error': (td_error ** 2).mean(),
                    'actor_loss': actor_loss.item(),
                    'entropy': entropy.clone().detach().mean()
                }
                # if epoch_i % self.config.log_per_updates == 0:
                log.update(stats)
                updates += 1

            checkpoint = {
                'model': gpt.state_dict(),
                'config': config,
                'epoch': epoch_i
            }

            # # Save checkpoint
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

                        zs = gpt.module.sample(x, cond=music_seq)

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
                gpt.module.freeze_drop()
            self.schedular.step()

    def eval(self):
        with torch.no_grad():
            vqvae = self.init_models[0].eval()
            gpt = self.init_models[1].train()

            config = self.config

            epoch_tested = config.testing.ckpt_epoch

            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation...")
            checkpoint = torch.load(ckpt_path)
            gpt.load_state_dict(checkpoint['model'], strict=False)
            gpt.eval()

            results = []
            quants_out = {}
            for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                # Prepare data
                music_seq, pose_seq = batch_eval
                music_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)

                quants = vqvae.module.encode(pose_seq)

                if isinstance(quants, tuple):
                    x = tuple(quants[i][0][:, :1] for i in range(len(quants)))
                else:
                    x = quants[0][:, :1]

                music_ds_rate = config.ds_rate if not hasattr(config, 'external_wav') else config.external_wav_rate
                music_seq = music_seq[:, :, :config.structure_generate.n_music // music_ds_rate].contiguous().float()
                b, t, c = music_seq.size()
                music_seq = music_seq.view(b, t // music_ds_rate, c * music_ds_rate)
                music_seq = music_seq[:, 1:]

                zs = gpt.module.sample(x, cond=music_seq)

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

    def visgt(self, ):
        config = self.config
        print("Visualizing ground truth")

        results = []
        random_id = 0  # np.random.randint(0, 1e4)

        for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
            # Prepare data
            pose_seq_eval = batch_eval

            results.append(pose_seq_eval)
        visualizeAndWrite(results, config, self.gtdir, self.dance_names, 0)

    def analyze_code(self, ):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        training_data = self.training_data
        all_quants = None

        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        random_id = 0  # np.random.randint(0, 1e4)

        for i_eval, batch_eval in enumerate(tqdm(self.training_data, desc='Generating Dance Poses')):
            # Prepare data
            pose_seq_eval = batch_eval.to(self.device)

            quants = model.module.encode(pose_seq_eval)[0].cpu().data.numpy()
            all_quants = np.append(all_quants, quants.reshape(-1)) if all_quants is not None else quants.reshape(-1)

        plt.hist(all_quants, bins=config.structure.l_bins, range=[0, config.structure.l_bins])

        log = datetime.datetime.now().strftime('%Y-%m-%d')
        plt.savefig(self.histdir1 + '/hist_epoch_' + str(epoch_tested) + '_%s.jpg' % log)  # 图片的存储
        plt.close()

    def sample(self, ):
        config = self.config
        print("Analyzing codebook")

        epoch_tested = config.testing.ckpt_epoch
        ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        model = self.model.eval()

        quants = {}

        results = []

        if hasattr(config, 'analysis_array') and config.analysis_array is not None:

            names = [str(ii) for ii in config.analysis_array]
            print(names)
            for ii in config.analysis_array:
                zs = [(ii * torch.ones((1, self.config.sample_code_length), device='cuda')).long()]

                pose_sample = model.module.decode(zs)
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii - 1, :3] + global_vel[:, iii - 1, :]

                quants[str(ii)] = zs[0].cpu().data.numpy()[0]

                results.append(pose_sample)
        else:
            names = ['rand_seq_' + str(ii) for ii in range(10)]
            for ii in range(10):
                zs = [torch.randint(0, self.config.structure.l_bins, size=(1, self.config.sample_code_length),
                                    device='cuda')]
                pose_sample = model.module.decode(zs)
                quants['rand_seq_' + str(ii)] = zs[0].cpu().data.numpy()[0]
                if config.global_vel:
                    global_vel = pose_sample[:, :, :3]
                    pose_sample[:, 0, :3] = 0
                    for iii in range(1, pose_sample.size(1)):
                        pose_sample[:, iii, :3] = pose_sample[:, iii - 1, :3] + global_vel[:, iii - 1, :]
                results.append(pose_sample)
        visualizeAndWrite(results, config, self.sampledir, names, epoch_tested, quants)


#     def _build(self):
#         config = self.config
#         self.start_epoch = 0
#         self._dir_setting()
#         self._build_model()
#         if not (hasattr(config, 'need_not_train_data') and config.need_not_train_data):
#             self._build_train_loader()
#         if not (hasattr(config, 'need_not_test_data') and config.need_not_train_data):
#             self._build_test_loader()
#         self._build_optimizer()
#
#     def _build_model(self):
#         """ Define Model """
#         config = self.config
#         if hasattr(config.structure, 'name') and hasattr(config.structure_generate, 'name'):
#             print(f'using {config.structure.name} and {config.structure_generate.name} ')
#             model_class = getattr(models, config.structure.name)
#             model = model_class(config.structure)
#
#             model_class2 = getattr(models, config.structure_generate.name)
#             model2 = model_class2(config.structure_generate)
#
#             model_reward = getattr(models, config.reward.name)
#             reward = model_reward(config.reward)
#         else:
#             raise NotImplementedError("Wrong Model Selection")
#
#         model = nn.DataParallel(model)
#         model2 = nn.DataParallel(model2)
#         dance_reward = nn.DataParallel(reward)
#         self.dance_reward = dance_reward.cuda()
#         self.model2 = model2.cuda()
#         self.model = model.cuda()
#
#     def _build_train_loader(self):
#
#         data = self.config.data
#         print("train with AIST++ dataset!")
#         external_wav_rate = self.config.ds_rate // self.config.external_wav_rate if hasattr(self.config,
#                                                                                             'external_wav_rate') else 1
#         external_wav_rate = self.config.music_relative_rate if hasattr(self.config,
#                                                                        'music_relative_rate') else external_wav_rate
#         train_music_data, train_dance_data, dance_names = load_data_aist(
#             data.train_dir, interval=data.seq_len, move=self.config.move if hasattr(self.config, 'move') else 64,
#             rotmat=self.config.rotmat,
#             external_wav=self.config.external_wav if hasattr(self.config, 'external_wav') else None,
#             external_wav_rate=external_wav_rate,
#             wav_padding=self.config.wav_padding * (
#                     self.config.ds_rate // self.config.music_relative_rate) if hasattr(self.config,
#                                                                                        'wav_padding') else 0)
#
#         # exit()
#         self.training_data = torch.utils.data.DataLoader(
#             MoDaSeq(train_music_data, train_dance_data),
#             num_workers=8,
#             batch_size=self.config.batch_size,
#             shuffle=True,
#             pin_memory=True
#         )
#         # self.dance_names = dance_names
#
#     def _build_test_loader(self):
#         config = self.config
#         data = self.config.data
#         print("test with AIST++ dataset!")
#         music_data, dance_data, dance_names = load_test_data_aist(
#             data.test_dir,
#             move=config.move,
#             rotmat=config.rotmat,
#             external_wav=config.external_wav if hasattr(self.config, 'external_wav') else None,
#             external_wav_rate=self.config.external_wav_rate if hasattr(self.config, 'external_wav_rate') else 1,
#             wav_padding=self.config.wav_padding * (
#                     self.config.ds_rate // self.config.music_relative_rate) if hasattr(self.config,
#                                                                                        'wav_padding') else 0)
#
#         self.test_loader = torch.utils.data.DataLoader(
#             MoDaSeq(music_data, dance_data),
#             batch_size=1,
#             shuffle=False
#         )
#         self.dance_names = dance_names
#
#     def _build_optimizer(self):
#         # model = nn.DataParallel(model).to(device)
#         config = self.config.optimizer
#         try:
#             optim = getattr(torch.optim, config.type)
#         except Exception:
#             raise NotImplementedError('not implemented optim method ' + config.type)
#
#         self.optimizer = optim(itertools.chain(self.model2.module.parameters(),
#                                                ),
#                                **config.kwargs)
#         self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)
#
#     def _dir_setting(self):
#         data = self.config.data
#         self.expname = self.config.expname
#         self.experiment_dir = os.path.join("../GMWM/", "experiments")
#         self.expdir = os.path.join(self.experiment_dir, self.expname)
#
#         if not os.path.exists(self.expdir):
#             os.mkdir(self.expdir)
#
#         self.visdir = os.path.join(self.expdir, "vis")  # -- imgs, videos, jsons
#         if not os.path.exists(self.visdir):
#             os.mkdir(self.visdir)
#
#         self.jsondir = os.path.join(self.visdir, "jsons")  # -- imgs, videos, jsons
#         if not os.path.exists(self.jsondir):
#             os.mkdir(self.jsondir)
#
#         self.histdir = os.path.join(self.visdir, "hist")  # -- imgs, videos, jsons
#         if not os.path.exists(self.histdir):
#             os.mkdir(self.histdir)
#
#         self.imgsdir = os.path.join(self.visdir, "imgs")  # -- imgs, videos, jsons
#         if not os.path.exists(self.imgsdir):
#             os.mkdir(self.imgsdir)
#
#         self.videodir = os.path.join(self.visdir, "videos")  # -- imgs, videos, jsons
#         if not os.path.exists(self.videodir):
#             os.mkdir(self.videodir)
#
#         self.ckptdir = os.path.join(self.expdir, "ckpt")
#         if not os.path.exists(self.ckptdir):
#             os.mkdir(self.ckptdir)
#
#         self.evaldir = os.path.join(self.expdir, "eval")
#         if not os.path.exists(self.evaldir):
#             os.mkdir(self.evaldir)
#
#         self.gtdir = os.path.join(self.expdir, "gt")
#         if not os.path.exists(self.gtdir):
#             os.mkdir(self.gtdir)
#
#         self.jsondir1 = os.path.join(self.evaldir, "jsons")  # -- imgs, videos, jsons
#         if not os.path.exists(self.jsondir1):
#             os.mkdir(self.jsondir1)
#
#         self.histdir1 = os.path.join(self.evaldir, "hist")  # -- imgs, videos, jsons
#         if not os.path.exists(self.histdir1):
#             os.mkdir(self.histdir1)
#
#         self.imgsdir1 = os.path.join(self.evaldir, "imgs")  # -- imgs, videos, jsons
#         if not os.path.exists(self.imgsdir1):
#             os.mkdir(self.imgsdir1)
#
#         self.videodir1 = os.path.join(self.evaldir, "videos")  # -- imgs, videos, jsons
#         if not os.path.exists(self.videodir1):
#             os.mkdir(self.videodir1)
#
#         self.sampledir = os.path.join(self.evaldir, "samples")  # -- imgs, videos, jsons
#         if not os.path.exists(self.sampledir):
#             os.mkdir(self.sampledir)
#
#         # self.ckptdir = os.path.join(self.expdir, "ckpt")
#         # if not os.path.exists(self.ckptdir):
#         #     os.mkdir(self.ckptdir)
#
#
def prepare_dataloader(music_data, dance_data, beat_data, batch_size, interval):
    modaac = MoDaSeqAC(music_data, dance_data, beat_data, interval)
    sampler = torch.utils.data.RandomSampler(modaac, replacement=True)

    data_loader = torch.utils.data.DataLoader(
        modaac,
        num_workers=8,
        batch_size=batch_size,
        # shuffle=True,
        sampler=sampler,
        pin_memory=True
        # collate_fn=paired_collate_fn,
    )

    return data_loader
