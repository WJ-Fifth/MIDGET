import torch
import torch.nn as nn
import torch.utils.data
from dataset.md_seq import MoDaSeq, paired_collate_fn

from utils.functional import str2bool, load_data, load_data_aist, check_data_distribution, visualizeAndWrite, \
    load_test_data_aist, load_test_data, dir_setting
import itertools
import models


def build(config):
    start_epoch = 0
    model_vqvae, model_gpt = _build_model(config)
    if not (hasattr(config, 'need_not_train_data') and config.need_not_train_data):
        training_data = _build_train_loader(config)
    else:
        training_data = None
    if not (hasattr(config, 'need_not_test_data') and config.need_not_test_data):
        test_loader, dance_names = _build_test_loader(config)
    else:
        test_loader, dance_names = None, None

    optimizer, schedular = _build_optimizer(config, model_vqvae, model_gpt)

    return (model_vqvae, model_gpt), training_data, test_loader, dance_names, optimizer, schedular


def _build_model(config):
    """ Define Model """
    if hasattr(config.structure, 'name') and hasattr(config.structure_generate, 'name'):
        print(f'using {config.structure.name} and {config.structure_generate.name} ')
        model_class_vqvae = getattr(models, config.structure.name)
        model_vqvae = model_class_vqvae(config.structure)

        model_class_gpt = getattr(models, config.structure_generate.name)
        model_gpt = model_class_gpt(config.structure_generate)
    else:
        raise NotImplementedError("Wrong Model Selection")

    model_vqvae = nn.DataParallel(model_vqvae)
    model_gpt = nn.DataParallel(model_gpt)
    return model_vqvae.cuda(), model_gpt.cuda()


def _build_train_loader(config):
    data = config.data
    if data.name == "aist":
        print("train with AIST++ dataset!")
        external_wav_rate = config.ds_rate // config.external_wav_rate if hasattr(config,
                                                                                  'external_wav_rate') else 1
        external_wav_rate = config.music_relative_rate if hasattr(config,
                                                                  'music_relative_rate') else external_wav_rate
        train_music_data, train_dance_data, _ = load_data_aist(
            data_dir=data.train_dir,
            interval=data.seq_len,
            move=config.move if hasattr(config, 'move') else 8,
            rotmat=config.rotmat,
            external_wav=config.external_wav if hasattr(config, 'external_wav') else None,
            external_wav_rate=external_wav_rate,
            wav_padding=config.wav_padding * (
                    config.ds_rate // config.music_relative_rate) if hasattr(config,
                                                                             'wav_padding') else 0)

    else:
        train_music_data, train_dance_data = load_data(
            data.train_dir,
            interval=data.seq_len,
            data_type=data.data_type)
    return prepare_dataloader(train_music_data, train_dance_data, config.batch_size)


def _build_test_loader(config):
    data = config.data
    print("test with AIST++ dataset!")
    music_data, dance_data, dance_names = load_test_data_aist(
        data.test_dir,
        move=config.move,
        rotmat=config.rotmat,
        external_wav=config.external_wav if hasattr(config, 'external_wav') else None,
        external_wav_rate=config.external_wav_rate if hasattr(config, 'external_wav_rate') else 1,
        wav_padding=config.wav_padding * (
                config.ds_rate // config.music_relative_rate) if hasattr(config,
                                                                         'wav_padding') else 0)

    test_loader = torch.utils.data.DataLoader(
        MoDaSeq(music_data, dance_data),
        batch_size=1,
        shuffle=False
        # collate_fn=paired_collate_fn,
    )
    dance_names = dance_names

    return test_loader, dance_names


def _build_optimizer(config, model_vqvae, model_gpt):
    # model = nn.DataParallel(model).to(device)
    config = config.optimizer
    try:
        optim = getattr(torch.optim, config.type)
    except Exception:
        raise NotImplementedError('not implemented optim method ' + config.type)

    optimizer = optim(itertools.chain(model_gpt.module.parameters(),
                                      ),
                      **config.kwargs)
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.schedular_kwargs)

    return optimizer, schedular


def prepare_dataloader(music_data, dance_data, batch_size):
    data_loader = torch.utils.data.DataLoader(
        MoDaSeq(music_data, dance_data),
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    return data_loader
