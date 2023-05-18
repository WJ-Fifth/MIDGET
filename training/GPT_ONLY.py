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
from torch.nn import functional as F
import numpy as np

warnings.filterwarnings('ignore')


class GPT_ONLY:
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

    def l1_loss_fn(self, x_target, x_pred):
        return torch.mean(torch.abs(x_pred - x_target))

    def l2_loss_fn(self, x_target, x_pred):
        return torch.mean(torch.square(x_pred - x_target))

    def train(self):
        gpt = self.init_models[1].train()

        config = self.config
        data = self.config.data
        training_data = self.training_data
        test_loader = self.test_loader
        optimizer = self.optimizer
        log = Logger(self.config, self.expdir)
        updates = 0
        losses = []
        total_losses = []

        if hasattr(config, 'init_weight') and config.init_weight is not None and config.init_weight != '':
            print('Use pretrained model!：', config.init_weight)
            # print(config.init_weight)
            checkpoint = torch.load(config.init_weight)
            gpt.load_state_dict(checkpoint['model'], strict=False)

        random.seed(config.seed)
        torch.manual_seed(config.seed)

        torch.cuda.manual_seed(config.seed)
        self.device = torch.device('cuda' if config.cuda else 'cpu')

        # Training Loop
        for epoch_i in range(201, config.epoch + 1):
            log.set_progress(epoch_i, len(training_data))

            for batch_i, batch in enumerate(training_data):
                music_seq, pose_seq = batch
                music_beats = music_seq[:, :, 53]

                music_seq = music_seq.to(self.device)
                pose_seq = pose_seq.to(self.device)
                music_beats = music_beats.to(self.device)
                pose_seq[:, :, :3] = 0

                optimizer.zero_grad()

                quants_input = pose_seq[:, :-8, :]
                quants_target = pose_seq[:, 8:, :]
                output, root = gpt(quants_input, music_seq, quants_target)

                ba_loss = self.BCE_Loss(output, music_beats[:, 8:])

                rec_loss = self.l2_loss_fn(output, quants_target)

                velocity_loss = self.l1_loss_fn(output[:, 1:] - output[:, :-1],
                                              quants_target[:, 1:] - quants_target[:, :-1])
                acceleration_loss = self.l1_loss_fn(output[:, 2:] + output[:, :-2] - 2 * output[:, 1:-1],
                                                  quants_target[:, 2:] +
                                                  quants_target[:, :-2] - 2 * quants_target[:, 1:-1])

                root_target = quants_target.float()[:, :, :3]

                root_rec_loss = self.l2_loss_fn(root_target, root)

                root_acc_loss = self.l1_loss_fn(root[:, 1:] - root[:, :-1], root_target[:, 1:] - root_target[:, :-1])

                loss = ba_loss + rec_loss + velocity_loss + acceleration_loss + 0.5 * (root_rec_loss + root_acc_loss)

                loss.backward()

                assert all(param.grad is not None for param in gpt.parameters()), \
                    'loss should depend differentiably on all neural network weights'

                # update parameters
                optimizer.step()

                stats = {'updates': updates, 'loss': loss.item(), 'BA_loss': ba_loss.item(),
                         'Rec_loss': rec_loss.item(), 'velocity_loss': velocity_loss.item(),
                         'acceleration_loss': acceleration_loss.item(),
                         "Root loss": 0.5 * (root_rec_loss + root_acc_loss).item()}

                log.update(stats)
                updates += 1

                losses.append(loss.item())
            total_loss = np.mean(losses)

            print("The total loss in single epoch:  %.4f" % total_loss)
            checkpoint = {'model': gpt.state_dict(), 'config': config, 'epoch': epoch_i, "loss": total_loss}

            # Save checkpoint
            if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
            # Eval
            self.schedular.step()

    def eval(self):
        with torch.no_grad():
            # vqvae = self.init_models[0].eval()
            gpt = self.init_models[1].eval()

            config = self.config
            epoch_tested = config.testing.ckpt_epoch

            ckpt_path = os.path.join(self.ckptdir, f"epoch_{epoch_tested}.pt")
            self.device = torch.device('cuda' if config.cuda else 'cpu')
            print("Evaluation...", ckpt_path)
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

                pose_seq[:, :, :3] = 0

                pose_sample = gpt.module.sample(pose_seq[:, :232, :], cond=music_seq,
                                                shift=config.sample_shift if hasattr(config, 'sample_shift') else None)

                if config.global_vel:
                    global_vel = pose_sample[:, :, :3].clone()
                    pose_sample[:, 0, :3] = 0
                    for i in range(1, pose_sample.size(1)):
                        pose_sample[:, i, :3] = pose_sample[:, i - 1, :3] + global_vel[:, i - 1, :]

                results.append(pose_sample)
                if isinstance(pose_sample, tuple):
                    quants_out[self.dance_names[i_eval]] = tuple(
                        pose_sample[ii][0].cpu().data.numpy()[0] for ii in range(len(pose_sample)))
                else:
                    quants_out[self.dance_names[i_eval]] = pose_sample[0].cpu().data.numpy()[0]

            visualizeAndWrite(results, config, self.evaldir, self.dance_names, epoch_tested, quants=None)
