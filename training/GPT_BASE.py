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

warnings.filterwarnings('ignore')


class GPT_BASE:
    def __init__(self, args):
        self.device = None
        self.config = args
        torch.backends.cudnn.benchmark = True
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
            # print(config.init_weight)
            checkpoint = torch.load(config.init_weight)
            gpt.load_state_dict(checkpoint['model'], strict=False)

        random.seed(config.seed)
        torch.manual_seed(config.seed)

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

                with torch.no_grad():
                    quants_pred = vqvae.module.encode(pose_seq)
                    if isinstance(quants_pred, tuple):
                        quants_input = tuple(
                            quants_pred[index][0][:, :-1].clone().detach() for index in range(len(quants_pred)))
                        quants_target = tuple(
                            quants_pred[index][0][:, 1:].clone().detach() for index in range(len(quants_pred)))
                    else:
                        quants = quants_pred[0]
                        quants_input = quants[:, :-1].clone().detach()
                        quants_target = quants[:, 1:].clone().detach()

                output, loss = gpt(quants_input, music_seq, quants_target)

                loss.backward()

                # update parameters
                optimizer.step()

                stats = {'updates': updates, 'loss': loss.item()}

                log.update(stats)
                updates += 1

            checkpoint = {'model': gpt.state_dict(), 'config': config, 'epoch': epoch_i}

            # Save checkpoint
            if epoch_i % config.save_per_epochs == 0:
                filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
                torch.save(checkpoint, filename)
            # Eval
            if epoch_i % config.test_freq == 0 or epoch_i == 1:
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

                        quants = vqvae.module.encode(pose_seq)

                        if isinstance(quants, tuple):
                            x = tuple(quants[i][0][:, :1] for i in range(len(quants)))
                        else:
                            x = quants[0][:, :1]

                        zs = gpt.module.sample(x, cond=music_seq,
                                               shift=config.sample_shift if hasattr(config, 'sample_shift') else None)
                        pose_sample = vqvae.module.decode(zs)

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
            print("Evaluation...")
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
