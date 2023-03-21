import numpy as np
import pickle
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import json
# kinetic, manual
import os
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt

music_root = './data/aistpp_test_full_wav'


def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        # print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]

        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]

        return beat_axis


def calc_db(keypoints, name):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats[0], len(kinetic_vel)


def BA(music_beats, motion_beats, sigma=3):
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats - bb) ** 2) / 2 / sigma ** 2)
    return ba / len(music_beats)


def BC(music_beats, motion_beats, sigma=3):
    bc = 0
    for motion_beat in motion_beats:
        bc += np.exp(-np.min((music_beats - motion_beat) ** 2) / 2 / sigma ** 2)
    return bc / len(motion_beats)


def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind] ** 2 / 2 / sigma ** 2)
        score_all.append(score)
    return sum(score_all) / len(score_all)


def calc_ba_score(root):
    bc_scores = []
    ba_scores = []
    best_score = 0.3
    lowest_score = 0.15
    best_motion = []
    lowest_motion = []

    for pkl in os.listdir(root):
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:, :]

        dance_beats, length = calc_db(joint3d, pkl)
        music_beats = get_mb(pkl.split('.')[0] + '.json', length)

        single_score = BA(music_beats, dance_beats)
        bc_score = BC(music_beats, dance_beats)

        ba_scores.append(single_score)
        bc_scores.append(bc_score)

        if single_score > best_score:
            best_motion.append(pkl)

        if single_score < lowest_score:
            lowest_motion.append(pkl)

    print("best")
    print(best_motion)
    print("lowest")
    print(lowest_motion)

    print(len(ba_scores))

    return np.mean(ba_scores), np.mean(bc_scores)


if __name__ == '__main__':
    # pred_root = './experiments/actor_critic/eval/pkl/ep000010'
    # pred_root = './experiments/motion_gpt_new/vis/pkl/ep000040'
    # pred_root = './experiments/motion_gpt_new/vis/pkl/ep000080'

    pred_root = './experiments/GPT_BA_BCE_1/eval/pkl/ep000040'
    # pred_root = 'experiments/motion_gpt_only_2/eval/pkl/ep000080'

    ba_scores, bc_scores = calc_ba_score(pred_root)

    print('Beat Align Score: %.4f' % ba_scores)
    print('Beat Consistency Score: %.4f' % bc_scores)
