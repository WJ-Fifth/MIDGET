""" This script handling the training process. """
import os
import random
import torch
import torch.utils.data

from utils.log import Logger
from utils.functional import visualizeAndWrite, dir_setting
import warnings
from tqdm import tqdm
import utils.build_util as build_util

import torchsummary

from models import gpt_model_ba, vqvae_root


from scipy.ndimage import gaussian_filter
import numpy as np

warnings.filterwarnings('ignore')

smpl_down = [0, 1, 2, 4, 5, 7, 8, 10, 11]
smpl_up = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

class GPT_BA:
    def __init__(self, args):
        self.device = None
        self.config = args
        torch.backends.cudnn.benchmark = True
        self.ckptdir, self.evaldir, self.gtdir, self.visdir, self.expdir = dir_setting(self.config)
        self.init_models, self.training_data, self.test_loader, self.dance_names, self.optimizer, self.schedular \
            = build_util.build(config=self.config)

    def BCE_Loss(self, pred_motions, music_beats):
        l2_distance = torch.mean(torch.square(pred_motions[:, :-1, :] - pred_motions[:, 1:, :]), dim=-1)
        motion_beats_prob = torch.exp(-l2_distance / 0.05 ** 2)

        extra = torch.zeros(pred_motions.shape[0])
        extra = extra.unsqueeze(1).to(self.device)
        motion_beats_prob = torch.cat((extra, motion_beats_prob), dim=1)

        bce_loss = torch.nn.BCELoss()
        loss = bce_loss(motion_beats_prob, music_beats.float())

        return loss

    def BA_Loss(self, pred_motions, music_beats):
        top_motion = pred_motions[:, :-1, :]
        next_motion = pred_motions[:, 1:, :]

        l2_distance = torch.mean(torch.square(top_motion - next_motion), dim=-1)
        motion_beats_prob = torch.exp(-l2_distance / 0.05 ** 2)

        extra = torch.zeros(top_motion.shape[0])
        extra = extra.unsqueeze(1).to(self.device)
        motion_beats_prob = torch.cat((extra, motion_beats_prob), dim=1)

        ba_loss = torch.mean(torch.square(music_beats - motion_beats_prob))
        return ba_loss

    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def train(self):
        config = self.config
        self.device = torch.device('cuda' if config.cuda else 'cpu')
        vqvae = self.init_models[0].eval()
        gpt = self.init_models[1].train()

        print(self.get_parameter_number(gpt))
        print(self.get_parameter_number(vqvae))
        exit()

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
            checkpoint = torch.load(config.init_weight)
            gpt.load_state_dict(checkpoint['model'], strict=False)

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        torch.cuda.manual_seed(config.seed)


        # Training Loop
        for epoch_i in range(61, config.epoch + 1):
            log.set_progress(epoch_i, len(training_data))

            for batch_i, batch in enumerate(training_data):

                music_seq, pose_seq = batch
                N, T, _ = pose_seq.shape
                music_beats = music_seq[:, :, 53]

                music_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)
                music_beats = music_beats.to(self.device)

                pose_seq[:, :, :3] = 0
                optimizer.zero_grad()

                with torch.no_grad():
                    quants_pred, up_codebook, down_codebook = vqvae.module.encode(pose_seq)
                    if isinstance(quants_pred, tuple):
                        quants_input = tuple(
                            quants_pred[index][0][:, :-1].clone().detach() for index in range(len(quants_pred)))
                        quants_target = tuple(
                            quants_pred[index][0][:, 1:].clone().detach() for index in range(len(quants_pred)))
                    else:
                        quants = quants_pred[0]
                        quants_input = quants[:, :-1].clone().detach()
                        quants_target = quants[:, 1:].clone().detach()

                # print("up joint codebook", up_codebook.shape)
                # print("down joint codebook", down_codebook.shape)

                output, joints_index, ce_loss = gpt(quants_input, music_seq, quants_target, (up_codebook, down_codebook))
                up_idx, down_idx = joints_index
                up_idx, down_idx = up_idx.permute(0, 2, 1).long().contiguous(), \
                    down_idx.permute(0, 2, 1).long().contiguous()

                pose_sample = vqvae.module.decode((up_idx, down_idx), output=output, bs_chunks=N)

                # x_quantised[0] = output[0] + (x_quantised[0] - output[0]).detach()
                # x_quantised[1] = output[1] + (x_quantised[1] - output[1]).detach()

                ba_loss = self.BCE_Loss(pose_sample, music_beats[:, 8:])

                loss = ba_loss + ce_loss

                loss.backward()


                # for param, name in zip(gpt.parameters(), gpt.named_parameters()):
                #     if param.grad is not None:
                #         print(name[0], ": Not None!")
                #     else:
                #         continue
                #         print(name[0])

                assert all(param.grad is not None for param in gpt.parameters()), \
                    'loss should depend differentiably on all neural network weights'

                # update parameters
                optimizer.step()

                stats = {'updates': updates, 'loss': loss.item(), 'CE_loss': ce_loss, 'BA_loss': ba_loss}

                log.update(stats)
                updates += 1

            checkpoint = {'model': gpt.state_dict(), 'config': config, 'epoch': epoch_i}

            # Save checkpoint
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
            # Eval
            if epoch_i % config.test_freq == 0 and epoch_i == 0:
                with torch.no_grad():
                    print("Evaluation...")
                    gpt.eval()
                    results = []
                    quants_out = {}
                    for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                        # Prepare data
                        music_seq, pose_seq = batch_eval

                        music_seq = music_seq.to(self.device)
                        pose_seq = pose_seq.to(self.device)

                        quants, up_codebook, down_codebook = vqvae.module.encode(pose_seq)

                        if isinstance(quants, tuple):
                            x = tuple(quants[i][0][:, :1] for i in range(len(quants)))
                        else:
                            x = quants[0][:, :1]

                        zs = gpt.module.sample(x, cond=music_seq,
                                               shift=config.sample_shift if hasattr(config, 'sample_shift') else None, codebooks=(up_codebook, down_codebook))
                        pose_sample = vqvae.module.decode(zs, output=(None, None))

                        # Calculate the position of the predicted pose relative to the global vector
                        if config.global_vel:
                            global_vel = pose_sample[:, :, :3].clone()
                            pose_sample[:, 0, :3] = 0
                            for t_d in range(1, pose_sample.size(1)):
                                pose_sample[:, t_d, :3] = pose_sample[:, t_d - 1, :3] + global_vel[:, t_d - 1, :]

                        # top-1 selection
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
            vqvae = self.init_models[0].eval()
            gpt = self.init_models[1].eval()

            config = self.config

            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)

            epoch_tested = config.testing.ckpt_epoch

            checkpoint = torch.load(config.vqvae_weight)
            vqvae.load_state_dict(checkpoint['model'], strict=False)

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation... ", ckpt_path)
            checkpoint = torch.load(ckpt_path)
            gpt.load_state_dict(checkpoint['model'])
            # gpt.eval()

            results = []
            # quants = {}
            quants_out = {}
            for i_eval, batch_eval in enumerate(tqdm(self.test_loader, desc='Generating Dance Poses')):
                # Prepare data
                music_seq, pose_seq = batch_eval
                music_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)

                quants, up_codebook, down_codebook = vqvae.module.encode(pose_seq)

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

                zs = gpt.module.sample(xs=x,
                                       cond=music_seq,
                                       shift=config.sample_shift if hasattr(config, 'sample_shift') else None,
                                       codebooks=(up_codebook, down_codebook))

                pose_sample = vqvae.module.decode(zs, output=(None, None))

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
            visualizeAndWrite(results, config, self.evaldir, self.dance_names, epoch_tested, quants_out, take_img=True)
